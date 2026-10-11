//! Conservative PE layout for separate recovered code and unwind metadata.
//!
//! Existing RVAs and raw offsets are never moved. New raw data is appended
//! after the complete input file, including its overlay. The overlay bytes
//! retain their offsets (they no longer necessarily form the file's suffix).
//! Two unused, zero-filled section-header slots are required; this module
//! refuses images which would need header relocation or section reshuffling.
//!
//! The caller owns directory contents, signature removal and final checksum.

use anyhow::{Context, Result, bail, ensure};

use crate::binary::pe::PeFile;

const SECTION_HEADER_SIZE: usize = 40;
const MAX_SECTIONS: usize = 96;
const CODE: u32 = 0x0000_0020;
const INITIALIZED_DATA: u32 = 0x0000_0040;
const EXECUTE: u32 = 0x2000_0000;
const READ: u32 = 0x4000_0000;
pub const GENERATED_CODE_CHARACTERISTICS: u32 = CODE | EXECUTE | READ;
pub const GENERATED_METADATA_CHARACTERISTICS: u32 = INITIALIZED_DATA | READ;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedSection {
    pub name: [u8; 8],
    pub rva: u32,
    pub raw_offset: u32,
    pub virtual_size: u32,
    pub raw_size: u32,
    pub characteristics: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SectionLayout {
    pub code: PlannedSection,
    pub metadata: PlannedSection,
    pub size_of_image: u32,
    pub size_of_code: u32,
    pub size_of_initialized_data: u32,
    pub base_of_code: u32,
    pub output_size: usize,
    source_size: usize,
    source_headers: Vec<u8>,
    header_offset: usize,
    number_of_sections_offset: usize,
    number_of_sections: u16,
    optional_header_offset: usize,
}

fn read_u16(data: &[u8], off: usize) -> Result<u16> {
    let end = off.checked_add(2).context("header offset overflow")?;
    Ok(u16::from_le_bytes(
        data.get(off..end)
            .context("truncated PE header")?
            .try_into()?,
    ))
}

fn read_u32(data: &[u8], off: usize) -> Result<u32> {
    let end = off.checked_add(4).context("header offset overflow")?;
    Ok(u32::from_le_bytes(
        data.get(off..end)
            .context("truncated PE header")?
            .try_into()?,
    ))
}

fn read_u64(data: &[u8], off: usize) -> Result<u64> {
    let end = off.checked_add(8).context("header offset overflow")?;
    Ok(u64::from_le_bytes(
        data.get(off..end)
            .context("truncated PE header")?
            .try_into()?,
    ))
}

fn align(value: u32, alignment: u32) -> Result<u32> {
    ensure!(
        alignment.is_power_of_two(),
        "alignment must be a nonzero power of two"
    );
    value
        .checked_add(alignment - 1)
        .map(|n| n & !(alignment - 1))
        .context("PE alignment overflow")
}

fn checked_size(size: usize, what: &str) -> Result<u32> {
    u32::try_from(size).with_context(|| format!("{what} exceeds PE 32-bit size limit"))
}

struct HeaderSpace {
    next_header: usize,
    count_offset: usize,
    new_count: u16,
    next_rva: u32,
}

fn validate_input(pe: &PeFile) -> Result<HeaderSpace> {
    ensure!(!pe.sections.is_empty(), "PE has no sections");
    ensure!(
        pe.sections.len() <= MAX_SECTIONS - 2,
        "adding two sections exceeds PE limit of 96 sections"
    );
    ensure!(
        (0x200..=0x10000).contains(&pe.file_alignment) && pe.file_alignment.is_power_of_two(),
        "unsupported FileAlignment {:#x}",
        pe.file_alignment
    );
    ensure!(
        pe.section_alignment >= 0x1000
            && pe.section_alignment >= pe.file_alignment
            && pe.section_alignment.is_power_of_two(),
        "unsupported SectionAlignment {:#x}; sub-page layouts require a separate mapping strategy",
        pe.section_alignment
    );
    let opt = pe.opt_header_offset;
    let pe_off = opt
        .checked_sub(24)
        .context("invalid optional-header offset")?;
    ensure!(
        read_u32(&pe.data, 0x3c)? as usize == pe_off
            && pe.data.get(pe_off..pe_off + 4) == Some(&b"PE\0\0"[..]),
        "PE header offsets do not match parsed image"
    );
    ensure!(
        read_u16(&pe.data, pe_off + 4)? == 0x8664 && read_u16(&pe.data, opt)? == 0x20b,
        "split-section layout supports AMD64 PE32+ only"
    );
    let count_offset = pe_off + 6;
    ensure!(
        read_u16(&pe.data, count_offset)? as usize == pe.sections.len(),
        "section count does not match parsed image"
    );
    ensure!(
        read_u32(&pe.data, pe_off + 12)? == 0 && read_u32(&pe.data, pe_off + 16)? == 0,
        "COFF symbol tables require a separate file-pointer preservation strategy"
    );
    let opt_size = read_u16(&pe.data, pe_off + 20)? as usize;
    ensure!(
        opt_size >= 112 && opt.checked_add(opt_size) == Some(pe.section_table_offset),
        "invalid PE32+ optional-header size"
    );
    let dirs = read_u32(&pe.data, opt + 108)? as usize;
    ensure!(
        dirs <= (opt_size - 112) / 8,
        "data directories exceed optional header"
    );
    ensure!(
        read_u64(&pe.data, opt + 24)? == pe.image_base
            && read_u32(&pe.data, opt + 32)? == pe.section_alignment
            && read_u32(&pe.data, opt + 36)? == pe.file_alignment
            && read_u32(&pe.data, opt + 56)? == pe.size_of_image
            && read_u32(&pe.data, opt + 60)? == pe.size_of_headers,
        "optional header differs from parsed layout"
    );
    let headers = pe.size_of_headers as usize;
    ensure!(
        headers <= pe.data.len() && pe.size_of_headers % pe.file_alignment == 0,
        "SizeOfHeaders is unbacked or misaligned"
    );
    let next_header = pe
        .section_table_offset
        .checked_add(pe.sections.len() * SECTION_HEADER_SIZE)
        .context("section-table offset overflow")?;
    let needed = next_header
        .checked_add(2 * SECTION_HEADER_SIZE)
        .context("section-table offset overflow")?;
    ensure!(
        needed <= headers,
        "insufficient section-header space for two sections; original RVAs will not be moved"
    );
    ensure!(
        pe.data
            .get(next_header..needed)
            .context("unbacked section-header slots")?
            .iter()
            .all(|b| *b == 0),
        "new section-header slots contain existing data"
    );
    for i in 0..dirs {
        let off = opt + 112 + i * 8;
        let start = read_u32(&pe.data, off)?;
        let size = read_u32(&pe.data, off + 4)?;
        if size == 0 {
            continue;
        }
        let end = start
            .checked_add(size)
            .context("data-directory range overflow")?;
        // Header RVAs equal file offsets. The security directory uses file
        // offsets everywhere, and must receive the same collision check.
        ensure!(
            end as usize <= next_header || start as usize >= needed,
            "section-header slots overlap data directory {i}"
        );
    }
    let mut raw_ranges = Vec::new();
    let mut virtual_ranges = Vec::new();
    let mut mapped_end = align(pe.size_of_headers, pe.section_alignment)?;
    for (i, sec) in pe.sections.iter().enumerate() {
        let h = pe.section_table_offset + i * SECTION_HEADER_SIZE;
        let raw_name = pe.data.get(h..h + 8).context("truncated section name")?;
        ensure!(
            String::from_utf8_lossy(raw_name).trim_end_matches('\0') == sec.name,
            "section {i} name differs from parsed layout"
        );
        ensure!(
            read_u32(&pe.data, h + 8)? == sec.virtual_size
                && read_u32(&pe.data, h + 12)? == sec.virtual_address
                && read_u32(&pe.data, h + 16)? == sec.raw_size
                && read_u32(&pe.data, h + 20)? == sec.raw_address
                && read_u32(&pe.data, h + 36)? == sec.characteristics,
            "section {i} differs from parsed layout"
        );
        ensure!(
            read_u32(&pe.data, h + 24)? == 0
                && read_u32(&pe.data, h + 28)? == 0
                && read_u16(&pe.data, h + 32)? == 0
                && read_u16(&pe.data, h + 34)? == 0,
            "section {i} has COFF relocation or line-number pointers; refusing header reuse"
        );
        ensure!(
            sec.virtual_address % pe.section_alignment == 0,
            "section {} has unaligned RVA",
            sec.name
        );
        let extent = sec.virtual_size.max(sec.raw_size);
        if extent != 0 {
            let end = sec
                .virtual_address
                .checked_add(extent)
                .context("existing section RVA overflow")?;
            ensure!(
                sec.virtual_address >= align(pe.size_of_headers, pe.section_alignment)?,
                "section {} overlaps mapped headers",
                sec.name
            );
            let mapped = align(end, pe.section_alignment)?;
            virtual_ranges.push((sec.virtual_address, mapped));
            mapped_end = mapped_end.max(mapped);
        }
        if sec.raw_size != 0 {
            ensure!(
                sec.raw_address % pe.file_alignment == 0 && sec.raw_size % pe.file_alignment == 0,
                "section {} has unaligned raw range",
                sec.name
            );
            let end = sec
                .raw_address
                .checked_add(sec.raw_size)
                .context("existing raw section overflow")?;
            ensure!(
                sec.raw_address >= pe.size_of_headers && end as usize <= pe.data.len(),
                "section {} overlaps headers or is not file backed",
                sec.name
            );
            raw_ranges.push((sec.raw_address, end));
        }
    }
    for (label, ranges) in [("raw", &mut raw_ranges), ("mapped", &mut virtual_ranges)] {
        ranges.sort_unstable();
        ensure!(
            ranges.windows(2).all(|w| w[0].1 <= w[1].0),
            "existing {label} section ranges overlap"
        );
    }
    ensure!(
        pe.size_of_image >= mapped_end && pe.size_of_image % pe.section_alignment == 0,
        "SizeOfImage does not cover the existing aligned image"
    );
    // Debug payloads may have only a file pointer (RVA zero), including to
    // header-resident zero bytes. Data-directory overlap checks alone would
    // miss those. Do not turn such payloads into new section headers.
    if dirs > 6 {
        let debug_dir = opt + 112 + 6 * 8;
        let debug_rva = read_u32(&pe.data, debug_dir)?;
        let debug_size = read_u32(&pe.data, debug_dir + 4)?;
        if debug_size != 0 {
            ensure!(debug_size % 28 == 0, "malformed debug directory size");
            let debug_end = debug_rva
                .checked_add(debug_size)
                .context("debug directory RVA overflow")?;
            let debug_raw = if debug_end <= pe.size_of_headers {
                debug_rva
            } else {
                let sec = pe
                    .sections
                    .iter()
                    .find(|s| {
                        debug_rva >= s.virtual_address
                            && debug_end <= s.virtual_address.saturating_add(s.raw_size)
                    })
                    .context("debug directory is not fully file backed")?;
                sec.raw_address
                    .checked_add(debug_rva - sec.virtual_address)
                    .context("debug directory raw offset overflow")?
            };
            for i in 0..debug_size / 28 {
                let off = debug_raw as usize + i as usize * 28;
                let len = read_u32(&pe.data, off + 16)?;
                let raw = read_u32(&pe.data, off + 24)?;
                if len == 0 {
                    continue;
                }
                let end = raw
                    .checked_add(len)
                    .context("debug payload file range overflow")?;
                ensure!(
                    raw != 0 && end as usize <= pe.data.len(),
                    "debug payload is not file backed"
                );
                ensure!(
                    end as usize <= next_header || raw as usize >= needed,
                    "new section headers overlap file-only debug payload"
                );
            }
        }
    }
    pe.image_base
        .checked_add(pe.size_of_image as u64)
        .context("image VA overflow")?;
    Ok(HeaderSpace {
        next_header,
        count_offset,
        new_count: u16::try_from(pe.sections.len() + 2)?,
        next_rva: pe.size_of_image,
    })
}

/// Fixed code RVA before emission. Requires room for both new headers even
/// though code and metadata lengths are not yet known.
pub fn next_code_rva(pe: &PeFile) -> Result<u32> {
    Ok(validate_input(pe)?.next_rva)
}

fn names(pe: &PeFile) -> Result<([u8; 8], [u8; 8])> {
    let used = |name: &str| pe.sections.iter().any(|s| s.name == name);
    if !used(".tvcode") && !used(".tvmeta") {
        return Ok((*b".tvcode\0", *b".tvmeta\0"));
    }
    for n in 1..=9999 {
        let code = format!(".tvc{n:04}");
        let meta = format!(".tvm{n:04}");
        if !used(&code) && !used(&meta) {
            return Ok((code.as_bytes().try_into()?, meta.as_bytes().try_into()?));
        }
    }
    bail!("generated section names exhausted")
}

/// Plan two fresh sections. Both payloads must be nonempty; callers that
/// emitted no functions should retain the input rather than append empty sections.
pub fn plan_sections(pe: &PeFile, code_size: usize, metadata_size: usize) -> Result<SectionLayout> {
    let space = validate_input(pe)?;
    ensure!(
        code_size != 0 && metadata_size != 0,
        "code and metadata payloads must both be nonempty"
    );
    let code_size = checked_size(code_size, "code payload")?;
    let metadata_size = checked_size(metadata_size, "metadata payload")?;
    let code_raw_size = align(code_size, pe.file_alignment)?;
    let metadata_raw_size = align(metadata_size, pe.file_alignment)?;
    let metadata_rva = align(
        space
            .next_rva
            .checked_add(code_size.max(code_raw_size))
            .context("code RVA range overflow")?,
        pe.section_alignment,
    )?;
    let size_of_image = align(
        metadata_rva
            .checked_add(metadata_size.max(metadata_raw_size))
            .context("metadata RVA range overflow")?,
        pe.section_alignment,
    )?;
    pe.image_base
        .checked_add(size_of_image as u64)
        .context("output image VA overflow")?;
    let code_raw = align(
        checked_size(pe.data.len(), "input file")?,
        pe.file_alignment,
    )?;
    let metadata_raw = code_raw
        .checked_add(code_raw_size)
        .context("code raw range overflow")?;
    let output_size = metadata_raw
        .checked_add(metadata_raw_size)
        .context("output file size overflow")? as usize;
    let mut size_of_code = code_raw_size;
    let mut size_of_initialized_data = metadata_raw_size;
    let mut base_of_code = space.next_rva;
    for sec in &pe.sections {
        if sec.characteristics & CODE != 0 {
            size_of_code = size_of_code
                .checked_add(sec.raw_size)
                .context("SizeOfCode overflow")?;
            if sec.virtual_size.max(sec.raw_size) != 0 {
                base_of_code = base_of_code.min(sec.virtual_address);
            }
        }
        if sec.characteristics & INITIALIZED_DATA != 0 {
            size_of_initialized_data = size_of_initialized_data
                .checked_add(sec.raw_size)
                .context("SizeOfInitializedData overflow")?;
        }
    }
    let (code_name, metadata_name) = names(pe)?;
    Ok(SectionLayout {
        code: PlannedSection {
            name: code_name,
            rva: space.next_rva,
            raw_offset: code_raw,
            virtual_size: code_size,
            raw_size: code_raw_size,
            characteristics: GENERATED_CODE_CHARACTERISTICS,
        },
        metadata: PlannedSection {
            name: metadata_name,
            rva: metadata_rva,
            raw_offset: metadata_raw,
            virtual_size: metadata_size,
            raw_size: metadata_raw_size,
            characteristics: GENERATED_METADATA_CHARACTERISTICS,
        },
        size_of_image,
        size_of_code,
        size_of_initialized_data,
        base_of_code,
        output_size,
        source_size: pe.data.len(),
        source_headers: pe.data[..pe.size_of_headers as usize].to_vec(),
        header_offset: space.next_header,
        number_of_sections_offset: space.count_offset,
        number_of_sections: space.new_count,
        optional_header_offset: pe.opt_header_offset,
    })
}

impl SectionLayout {
    /// Write payloads and layout headers. The existing image prefix differs
    /// only in the two previously-unused section headers and aggregate fields.
    /// Exception/security directories and checksum remain caller-owned.
    pub fn write_sections(&self, pe: &PeFile, code: &[u8], metadata: &[u8]) -> Result<Vec<u8>> {
        // Replanning also rejects an altered plan, different base image/header,
        // stale sizes or accidentally swapped metadata/code payload lengths.
        ensure!(
            *self == plan_sections(pe, code.len(), metadata.len())?,
            "section layout no longer matches input or payload sizes"
        );
        let mut out = Vec::new();
        out.try_reserve_exact(self.output_size)
            .context("allocating split-section output")?;
        out.extend_from_slice(&pe.data);
        out.resize(self.output_size, 0);
        let code_at = self.code.raw_offset as usize;
        out[code_at..code_at + self.code.raw_size as usize].fill(0xcc);
        out[code_at..code_at + code.len()].copy_from_slice(code);
        let meta_at = self.metadata.raw_offset as usize;
        out[meta_at..meta_at + metadata.len()].copy_from_slice(metadata);
        for (i, section) in [&self.code, &self.metadata].iter().enumerate() {
            let off = self.header_offset + i * SECTION_HEADER_SIZE;
            out[off..off + SECTION_HEADER_SIZE].fill(0);
            out[off..off + 8].copy_from_slice(&section.name);
            for (delta, value) in [
                (8, section.virtual_size),
                (12, section.rva),
                (16, section.raw_size),
                (20, section.raw_offset),
                (36, section.characteristics),
            ] {
                out[off + delta..off + delta + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
        out[self.number_of_sections_offset..self.number_of_sections_offset + 2]
            .copy_from_slice(&self.number_of_sections.to_le_bytes());
        for (delta, value) in [
            (4, self.size_of_code),
            (8, self.size_of_initialized_data),
            (20, self.base_of_code),
            (56, self.size_of_image),
        ] {
            let off = self.optional_header_offset + delta;
            out[off..off + 4].copy_from_slice(&value.to_le_bytes());
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put16(data: &mut [u8], off: usize, n: u16) {
        data[off..off + 2].copy_from_slice(&n.to_le_bytes());
    }
    fn put32(data: &mut [u8], off: usize, n: u32) {
        data[off..off + 4].copy_from_slice(&n.to_le_bytes());
    }
    fn fixture(headers: u32, overlay: &[u8]) -> PeFile {
        let mut data = vec![0u8; headers as usize + 3 * 0x200];
        data[..2].copy_from_slice(b"MZ");
        put32(&mut data, 0x3c, 0x80);
        data[0x80..0x84].copy_from_slice(b"PE\0\0");
        put16(&mut data, 0x84, 0x8664);
        put16(&mut data, 0x86, 3);
        put16(&mut data, 0x94, 0xf0);
        let opt = 0x98;
        put16(&mut data, opt, 0x20b);
        put32(&mut data, opt + 4, 0x200);
        put32(&mut data, opt + 8, 0x400);
        put32(&mut data, opt + 16, 0x1000);
        put32(&mut data, opt + 20, 0x1000);
        data[opt + 24..opt + 32].copy_from_slice(&0x180000000u64.to_le_bytes());
        put32(&mut data, opt + 32, 0x1000);
        put32(&mut data, opt + 36, 0x200);
        put32(&mut data, opt + 56, 0x4000);
        put32(&mut data, opt + 60, headers);
        put32(&mut data, opt + 108, 16);
        for (i, (name, chars)) in [
            (*b".text\0\0\0", CODE | EXECUTE | READ),
            (*b".data\0\0\0", INITIALIZED_DATA | READ | 0x80000000),
            (*b".reloc\0\0", INITIALIZED_DATA | READ | 0x02000000),
        ]
        .iter()
        .enumerate()
        {
            let off = 0x188 + i * 40;
            data[off..off + 8].copy_from_slice(name);
            put32(&mut data, off + 8, 0x180);
            put32(&mut data, off + 12, 0x1000 * (i as u32 + 1));
            put32(&mut data, off + 16, 0x200);
            put32(&mut data, off + 20, headers + i as u32 * 0x200);
            put32(&mut data, off + 36, *chars);
            let raw = headers as usize + i * 0x200;
            data[raw..raw + 0x200].fill(0x20 + i as u8);
        }
        data.extend_from_slice(overlay);
        PeFile::parse(data).unwrap()
    }

    #[test]
    fn separate_permissions_overlay_and_aggregate_sizes() {
        let pe = fixture(0x400, b"overlay must keep its file offset\0\xff");
        let layout = plan_sections(&pe, 17, 523).unwrap();
        assert_eq!(next_code_rva(&pe).unwrap(), layout.code.rva);
        assert_eq!(layout.code.characteristics, 0x60000020);
        assert_eq!(layout.metadata.characteristics, 0x40000040);
        assert_eq!(layout.code.raw_offset, 0xc00);
        assert_eq!(layout.code.rva, 0x4000);
        assert_eq!(layout.metadata.rva, 0x5000);
        assert_eq!(layout.size_of_image, 0x6000);
        let code = [0x90; 17];
        let metadata = [0x34; 523];
        let out = layout.write_sections(&pe, &code, &metadata).unwrap();
        assert_eq!(
            &out[pe.size_of_headers as usize..pe.data.len()],
            &pe.data[pe.size_of_headers as usize..]
        );
        assert_eq!(&out[0xc00..0xc11], &code);
        assert!(out[0xc11..0xe00].iter().all(|b| *b == 0xcc));
        assert_eq!(&out[0xe00..0xe00 + 523], &metadata);
        let reparsed = PeFile::parse(out).unwrap();
        assert_eq!(reparsed.sections.len(), 5);
        assert_eq!(reparsed.sections[3].name, ".tvcode");
        assert_eq!(reparsed.sections[4].name, ".tvmeta");
        assert_eq!(read_u32(&reparsed.data, 0x98 + 4).unwrap(), 0x400);
        assert_eq!(read_u32(&reparsed.data, 0x98 + 8).unwrap(), 0x800);
        assert_eq!(
            reparsed.sections[2].characteristics,
            pe.sections[2].characteristics
        );
        assert_eq!(
            reparsed.sections[2].virtual_size,
            pe.sections[2].virtual_size
        );
    }

    #[test]
    fn two_incremental_appends_keep_every_old_section() {
        let pe = fixture(0x400, b"overlay");
        let first = plan_sections(&pe, 19, 37).unwrap();
        let first_pe =
            PeFile::parse(first.write_sections(&pe, &[0xc3; 19], &[0x41; 37]).unwrap()).unwrap();
        let second = plan_sections(&first_pe, 4097, 3).unwrap();
        let final_pe = PeFile::parse(
            second
                .write_sections(&first_pe, &vec![0x90; 4097], &[1, 2, 3])
                .unwrap(),
        )
        .unwrap();
        assert_eq!(final_pe.sections.len(), 7);
        assert_eq!(final_pe.sections[5].name, ".tvc0001");
        assert_eq!(final_pe.sections[6].name, ".tvm0001");
        assert_eq!(second.code.rva, first.size_of_image);
        assert_eq!(second.metadata.rva, second.code.rva + 0x2000);
        for (a, b) in first_pe.sections.iter().zip(&final_pe.sections) {
            assert_eq!(
                (
                    a.virtual_address,
                    a.virtual_size,
                    a.raw_address,
                    a.raw_size,
                    a.characteristics
                ),
                (
                    b.virtual_address,
                    b.virtual_size,
                    b.raw_address,
                    b.raw_size,
                    b.characteristics
                )
            );
            assert_eq!(first_pe.section_bytes(a), final_pe.section_bytes(b));
        }
    }

    #[test]
    fn missing_or_occupied_header_space_is_rejected() {
        let pe = fixture(0x200, &[]);
        assert!(
            plan_sections(&pe, 1, 1)
                .unwrap_err()
                .to_string()
                .contains("section-header space")
        );
        let mut pe = fixture(0x400, &[]);
        pe.data[0x200] = 0xaa;
        assert!(
            plan_sections(&pe, 1, 1)
                .unwrap_err()
                .to_string()
                .contains("existing data")
        );
    }

    #[test]
    fn header_directory_payload_is_not_overwritten_even_when_zero() {
        let mut pe = fixture(0x400, &[]);
        put32(&mut pe.data, 0x98 + 112 + 6 * 8, 0x218);
        put32(&mut pe.data, 0x98 + 116 + 6 * 8, 8);
        assert!(
            plan_sections(&pe, 1, 1)
                .unwrap_err()
                .to_string()
                .contains("data directory 6")
        );
    }

    #[test]
    fn unsupported_alignments_and_overflows_reject_before_allocation() {
        let pe = fixture(0x400, &[]);
        assert!(plan_sections(&pe, u32::MAX as usize, 1).is_err());
        assert!(plan_sections(&pe, 1, u32::MAX as usize).is_err());
        assert!(plan_sections(&pe, 0, 1).is_err());
        for alignment in [0, 3, 0x100] {
            let mut bad = fixture(0x400, &[]);
            bad.file_alignment = alignment;
            assert!(plan_sections(&bad, 1, 1).is_err());
        }
        let mut bad = fixture(0x400, &[]);
        bad.size_of_image = 0xfffff000;
        put32(&mut bad.data, 0x98 + 56, bad.size_of_image);
        assert!(plan_sections(&bad, 1, 1).is_err());
        let mut bad = fixture(0x400, &[]);
        bad.image_base = u64::MAX - 0x8000;
        bad.data[0x98 + 24..0x98 + 32].copy_from_slice(&bad.image_base.to_le_bytes());
        assert!(plan_sections(&bad, 0x8000, 1).is_err());
    }

    #[test]
    fn writer_rejects_stale_or_tampered_plan_and_wrong_payload_length() {
        let pe = fixture(0x400, &[]);
        let layout = plan_sections(&pe, 1, 1).unwrap();
        assert!(layout.write_sections(&pe, &[1, 2], &[3]).is_err());
        let mut altered = layout.clone();
        altered.metadata.characteristics |= EXECUTE;
        assert!(altered.write_sections(&pe, &[1], &[3]).is_err());
        let mut changed = fixture(0x400, &[]);
        changed.data[0x50] = 1;
        assert!(layout.write_sections(&changed, &[1], &[3]).is_err());
    }

    #[test]
    fn writer_only_changes_declared_layout_header_bytes_in_original_prefix() {
        let pe = fixture(0x400, b"file overlay");
        let plan = plan_sections(&pe, 7, 13).unwrap();
        let out = plan.write_sections(&pe, &[0x90; 7], &[0x77; 13]).unwrap();
        let allowed = [
            (
                plan.number_of_sections_offset,
                plan.number_of_sections_offset + 2,
            ),
            (plan.header_offset, plan.header_offset + 80),
            (pe.opt_header_offset + 4, pe.opt_header_offset + 12),
            (pe.opt_header_offset + 20, pe.opt_header_offset + 24),
            (pe.opt_header_offset + 56, pe.opt_header_offset + 60),
        ];
        for (i, (before, after)) in pe.data.iter().zip(&out).enumerate() {
            if before != after {
                assert!(
                    allowed.iter().any(|&(lo, hi)| lo <= i && i < hi),
                    "unexpected changed offset {i:#x}"
                );
            }
        }
    }

    #[test]
    fn symbol_line_and_file_only_debug_data_are_not_silent_header_scratch() {
        let mut symbols = fixture(0x400, &[]);
        put32(&mut symbols.data, 0x80 + 12, 0x220);
        assert!(
            plan_sections(&symbols, 1, 1)
                .unwrap_err()
                .to_string()
                .contains("COFF symbol")
        );
        let mut lines = fixture(0x400, &[]);
        put32(&mut lines.data, 0x188 + 28, 0x220);
        assert!(
            plan_sections(&lines, 1, 1)
                .unwrap_err()
                .to_string()
                .contains("line-number")
        );
        let mut debug = fixture(0x400, &[]);
        put32(&mut debug.data, 0x98 + 112 + 6 * 8, 0x2000);
        put32(&mut debug.data, 0x98 + 116 + 6 * 8, 28);
        debug.data[0x600..0x61c].fill(0);
        put32(&mut debug.data, 0x600 + 16, 16);
        put32(&mut debug.data, 0x600 + 24, 0x218);
        assert!(
            plan_sections(&debug, 1, 1)
                .unwrap_err()
                .to_string()
                .contains("file-only debug")
        );
    }

    #[test]
    fn every_public_plan_geometry_mutation_is_rejected() {
        let pe = fixture(0x400, &[]);
        let plan = plan_sections(&pe, 1, 1).unwrap();
        let mut mutants = Vec::new();
        let mut p = plan.clone();
        p.code.rva = 0x1000;
        mutants.push(p);
        let mut p = plan.clone();
        p.metadata.raw_offset = p.code.raw_offset;
        mutants.push(p);
        let mut p = plan.clone();
        p.code.raw_size = u32::MAX;
        mutants.push(p);
        let mut p = plan.clone();
        p.metadata.virtual_size = u32::MAX;
        mutants.push(p);
        let mut p = plan.clone();
        p.code.name = *b".text\0\0\0";
        mutants.push(p);
        let mut p = plan.clone();
        p.size_of_image = 0;
        mutants.push(p);
        let mut p = plan.clone();
        p.size_of_code = 0;
        mutants.push(p);
        let mut p = plan.clone();
        p.size_of_initialized_data = 0;
        mutants.push(p);
        let mut p = plan.clone();
        p.base_of_code = 0;
        mutants.push(p);
        let mut p = plan.clone();
        p.output_size = 1;
        mutants.push(p);
        for changed in mutants {
            assert!(changed.write_sections(&pe, &[0xc3], &[1]).is_err());
        }
    }
}
