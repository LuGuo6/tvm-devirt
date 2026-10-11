//! Conservative AMD64 exception-metadata inspection, not handler translation.
//!
//! `Unprotected` means no language handler is described for the queried PCs. It
//! does not prove runtime equivalence, and says nothing about their callees.
//! Unsupported or malformed metadata is `Unknown`, never absence of a handler.
//! Version 1 unwind operations, CHAININFO, and low-bit indirect RUNTIME_FUNCTION
//! entries are supported. Version 2/3 semantics are deliberately not guessed.

use super::pe::PeFile;
use std::cell::{Cell, RefCell};
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoverageKind {
    Unprotected,
    Handler,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandlerInfo {
    /// The queried table record, rather than only the terminal chained record.
    pub runtime_begin_rva: u32,
    pub runtime_end_rva: u32,
    pub unwind_rva: u32,
    pub handler_rva: u32,
    /// EHANDLER=1, UHANDLER=2; both bits may be set.
    pub flags: u8,
    /// Start of opaque language-specific data; its schema is not inferred.
    pub language_data_rva: u32,
    /// UNWIND_INFO RVAs followed from this record to the handler.
    pub unwind_chain: Vec<u32>,
    /// Indirect RUNTIME_FUNCTION RVAs followed through UnwindData's low bit.
    pub indirect_records: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Coverage {
    pub kind: CoverageKind,
    pub handlers: Vec<HandlerInfo>,
    pub issues: Vec<String>,
    pub runtime_ranges: Vec<(u32, u32)>,
}

impl Coverage {
    fn unprotected() -> Self {
        Self {
            kind: CoverageKind::Unprotected,
            handlers: Vec::new(),
            issues: Vec::new(),
            runtime_ranges: Vec::new(),
        }
    }

    fn unknown(reason: impl Into<String>) -> Self {
        let mut result = Self::unprotected();
        result.kind = CoverageKind::Unknown;
        result.issues.push(reason.into());
        result
    }

    fn merge(&mut self, other: &Self) {
        self.handlers.extend(other.handlers.iter().cloned());
        self.issues.extend(other.issues.iter().cloned());
        self.runtime_ranges
            .extend(other.runtime_ranges.iter().copied());
        self.kind = if self.kind == CoverageKind::Unknown || other.kind == CoverageKind::Unknown {
            CoverageKind::Unknown
        } else if self.kind == CoverageKind::Handler || other.kind == CoverageKind::Handler {
            CoverageKind::Handler
        } else {
            CoverageKind::Unprotected
        };
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeFunction {
    pub record_rva: u32,
    pub begin_rva: u32,
    pub end_rva: u32,
    pub unwind_data: u32,
    pub coverage: Coverage,
}

#[derive(Debug)]
pub struct ExceptionTable {
    pub image_base: u64,
    pub size_of_image: u32,
    pub directory_rva: u32,
    pub directory_size: u32,
    pub records: Vec<RuntimeFunction>,
    /// A malformed directory or ambiguous PE mapping invalidates all lookups.
    pub global_errors: Vec<String>,
    /// Sorted, merged half-open RVA ranges which entry/relay patches must
    /// preserve. Structural PE headers are included only when referenced as
    /// exception metadata. Opaque/unsupported payloads reserve their container tail.
    pub metadata_ranges: Vec<(u32, u32)>,
    executable_raw_ranges: Vec<(u32, u32)>,
}

impl ExceptionTable {
    pub fn parse(pe: &PeFile) -> Self {
        let mut result = Self {
            image_base: pe.image_base,
            size_of_image: pe.size_of_image,
            directory_rva: 0,
            directory_size: 0,
            records: Vec::new(),
            global_errors: Vec::new(),
            metadata_ranges: Vec::new(),
            executable_raw_ranges: Vec::new(),
        };
        let map = ImageMap {
            pe,
            metadata: RefCell::new(Vec::new()),
            metadata_cursor: Cell::new(None),
        };
        if let Err(error) = map.validate() {
            result.global_errors.push(error);
            return result;
        }
        for section in &pe.sections {
            if section.is_executable() && section.raw_size != 0 {
                result.executable_raw_ranges.push((
                    section.virtual_address,
                    section.virtual_address + section.raw_size,
                ));
            }
        }
        result.executable_raw_ranges.sort_unstable();
        let directory = match map.exception_directory() {
            Ok(directory) => directory,
            Err(error) => {
                result.global_errors.push(error);
                return result;
            }
        };
        let Some((rva, size)) = directory else {
            result.metadata_ranges = map.metadata.into_inner();
            return result;
        };
        result.directory_rva = rva;
        result.directory_size = size;
        if rva % 4 != 0 || size % 12 != 0 {
            result.global_errors.push(
                "exception directory is not DWORD-aligned or contains a partial RUNTIME_FUNCTION"
                    .into(),
            );
            return result;
        }
        let data = match map.read(rva, size) {
            Ok(data) => data,
            Err(error) => {
                result
                    .global_errors
                    .push(format!("exception directory: {error}"));
                return result;
            }
        };
        map.metadata.borrow_mut().push((rva, rva + size));
        let mut previous_end = 0;
        for (index, bytes) in data.chunks_exact(12).enumerate() {
            let record_rva = rva + index as u32 * 12;
            let (begin_rva, end_rva, unwind_data) = runtime_fields(bytes);
            if let Err(error) = map.code_range(begin_rva, end_rva) {
                result
                    .global_errors
                    .push(format!("runtime record {record_rva:#x}: {error}"));
            }
            if index != 0 && begin_rva < previous_end {
                result.global_errors.push(format!(
                    "runtime record {record_rva:#x}: unsorted, duplicate or overlapping range"
                ));
            }
            previous_end = end_rva;
            result.records.push(RuntimeFunction {
                record_rva,
                begin_rva,
                end_rva,
                unwind_data,
                coverage: Coverage::unprotected(),
            });
        }
        if !result.global_errors.is_empty() {
            return result;
        }
        for record in &mut result.records {
            record.coverage = resolve(&map, record);
            record
                .coverage
                .runtime_ranges
                .push((record.begin_rva, record.end_rva));
        }
        let mut ranges = map.metadata.into_inner();
        ranges.sort_unstable();
        for (lo, hi) in ranges {
            if let Some(last) = result.metadata_ranges.last_mut()
                && lo <= last.1
            {
                last.1 = last.1.max(hi);
            } else {
                result.metadata_ranges.push((lo, hi));
            }
        }
        result
    }

    /// First preserved exception-metadata interval overlapping [begin,end).
    /// VAs are used to match entry/relay patch APIs. Invalid queries or a table
    /// with global mapping errors fail closed and return the query as evidence.
    pub fn patch_metadata_overlap(&self, begin: u64, end: u64) -> Option<(u64, u64)> {
        let image_end = self.image_base.checked_add(u64::from(self.size_of_image));
        if begin >= end
            || begin < self.image_base
            || image_end.is_none_or(|limit| end > limit)
            || !self.global_errors.is_empty()
        {
            return Some((begin, end));
        }
        let begin_rva = (begin - self.image_base) as u32;
        let end_rva = (end - self.image_base) as u32;
        let index = self
            .metadata_ranges
            .partition_point(|range| range.1 <= begin_rva);
        self.metadata_ranges
            .get(index)
            .filter(|range| range.0 < end_rva)
            .map(|&(lo, hi)| {
                (
                    self.image_base + u64::from(lo),
                    self.image_base + u64::from(hi),
                )
            })
    }

    pub fn coverage_va(&self, va: u64) -> Coverage {
        let Some(end) = va.checked_add(1) else {
            return Coverage::unknown("query address overflow");
        };
        self.coverage_range_va(va, end)
    }

    /// Query [begin, end), including every intersecting record and interior PC.
    /// An empty/reversed range or an unbacked/non-executable gap is Unknown.
    pub fn coverage_range_va(&self, begin: u64, end: u64) -> Coverage {
        let Some(begin) = begin
            .checked_sub(self.image_base)
            .and_then(|x| u32::try_from(x).ok())
        else {
            return Coverage::unknown("query begins outside the PE image");
        };
        let Some(end) = end
            .checked_sub(self.image_base)
            .and_then(|x| u32::try_from(x).ok())
        else {
            return Coverage::unknown("query ends outside the PE image");
        };
        self.coverage_range_rva(begin, end)
    }

    pub fn coverage_rva(&self, rva: u32) -> Coverage {
        match rva.checked_add(1) {
            Some(end) => self.coverage_range_rva(rva, end),
            None => Coverage::unknown("query RVA overflow"),
        }
    }

    /// Sorted half-open VA intervals that cannot be translated with stack-only
    /// unwind metadata. Non-code/unbacked gaps are included as Unknown. VAs
    /// outside the image must be rejected separately by the caller.
    pub fn unsafe_ranges(&self) -> Vec<(u64, u64, String)> {
        let Some(image_end) = self.image_base.checked_add(u64::from(self.size_of_image)) else {
            return vec![(0, u64::MAX, "invalid image extent".into())];
        };
        if !self.global_errors.is_empty() {
            return vec![(self.image_base, image_end, self.global_errors.join("; "))];
        }
        let mut result = Vec::new();
        let mut cursor = 0;
        for &(lo, hi) in &self.executable_raw_ranges {
            if cursor < lo {
                result.push((
                    self.image_base + u64::from(cursor),
                    self.image_base + u64::from(lo),
                    "Unknown: non-executable or non-file-backed bytes".into(),
                ));
            }
            cursor = hi;
        }
        if cursor < self.size_of_image {
            result.push((
                self.image_base + u64::from(cursor),
                image_end,
                "Unknown: non-executable or non-file-backed bytes".into(),
            ));
        }
        for record in &self.records {
            let reason = match record.coverage.kind {
                CoverageKind::Unprotected => continue,
                CoverageKind::Handler => format!(
                    "Handler: original language exception/termination handler covers {:#x}..{:#x}",
                    record.begin_rva, record.end_rva
                ),
                CoverageKind::Unknown => format!("Unknown: {}", record.coverage.issues.join("; ")),
            };
            result.push((
                self.image_base + u64::from(record.begin_rva),
                self.image_base + u64::from(record.end_rva),
                reason,
            ));
        }
        result.sort_unstable_by_key(|range| range.0);
        result
    }

    pub fn coverage_range_rva(&self, begin: u32, end: u32) -> Coverage {
        if !self.global_errors.is_empty() {
            let mut result = Coverage::unknown("invalid PE exception table or image mapping");
            result.issues.extend(self.global_errors.iter().cloned());
            return result;
        }
        if begin >= end || end > self.size_of_image {
            return Coverage::unknown("empty, reversed or out-of-image query range");
        }
        let mut cursor = begin;
        for &(lo, hi) in &self.executable_raw_ranges {
            if hi <= cursor {
                continue;
            }
            if lo > cursor {
                break;
            }
            cursor = hi.min(end);
            if cursor == end {
                break;
            }
        }
        if cursor != end {
            return Coverage::unknown("query includes non-executable or non-file-backed bytes");
        }
        let mut result = Coverage::unprotected();
        let first = self
            .records
            .partition_point(|record| record.end_rva <= begin);
        for record in &self.records[first..] {
            if record.begin_rva >= end {
                break;
            }
            result.merge(&record.coverage);
        }
        result
    }
}

struct ImageMap<'a> {
    pe: &'a PeFile,
    metadata: RefCell<Vec<(u32, u32)>>,
    metadata_cursor: Cell<Option<u32>>,
}

impl<'a> ImageMap<'a> {
    fn read_metadata(&self, rva: u32, size: u32) -> Result<&'a [u8], String> {
        self.metadata_cursor.set(Some(rva));
        let bytes = self.read(rva, size)?;
        self.metadata.borrow_mut().push((rva, rva + size));
        Ok(bytes)
    }

    /// Neither language-specific handler data nor an unsupported unwind format
    /// has a proven length here. Preserve the remainder of its original mapped
    /// container instead of guessing a short length from a header or zero bytes.
    fn preserve_metadata_tail(&self, rva: u32) {
        // RVA zero is a missing pointer, not a claim that DOS/PE headers carry
        // an unknown unwind structure. Its record still remains Unknown.
        if rva == 0 {
            return;
        }
        if rva < self.pe.size_of_headers {
            self.metadata
                .borrow_mut()
                .push((rva, self.pe.size_of_headers));
            return;
        }
        for section in &self.pe.sections {
            let end = section.virtual_address + section.virtual_size.max(section.raw_size);
            if section.virtual_address <= rva && rva < end {
                self.metadata.borrow_mut().push((rva, end));
                return;
            }
        }
    }

    fn validate(&self) -> Result<(), String> {
        let pe = self.pe;
        if pe.size_of_headers == 0
            || pe.size_of_headers > pe.size_of_image
            || pe.size_of_headers as usize > pe.data.len()
            || pe
                .image_base
                .checked_add(u64::from(pe.size_of_image))
                .is_none()
        {
            return Err("invalid image/header extent".into());
        }
        let mut virtual_ranges = vec![(0, pe.size_of_headers)];
        let mut raw_ranges = vec![(0u64, u64::from(pe.size_of_headers))];
        for section in &pe.sections {
            let extent = section.virtual_size.max(section.raw_size);
            if extent == 0 {
                continue;
            }
            let end = section
                .virtual_address
                .checked_add(extent)
                .ok_or("section RVA overflow")?;
            if end > pe.size_of_image {
                return Err("section exceeds SizeOfImage".into());
            }
            virtual_ranges.push((section.virtual_address, end));
            if section.raw_size != 0 {
                let end = u64::from(section.raw_address) + u64::from(section.raw_size);
                if end > pe.data.len() as u64 {
                    return Err("section raw extent is truncated".into());
                }
                raw_ranges.push((u64::from(section.raw_address), end));
            }
        }
        virtual_ranges.sort_unstable();
        raw_ranges.sort_unstable();
        if virtual_ranges.windows(2).any(|pair| pair[0].1 > pair[1].0)
            || raw_ranges.windows(2).any(|pair| pair[0].1 > pair[1].0)
        {
            return Err("ambiguous overlapping PE section/header mappings".into());
        }
        Ok(())
    }

    fn exception_directory(&self) -> Result<Option<(u32, u32)>, String> {
        let pe = self.pe;
        let read16 = |offset: usize| -> Result<u16, String> {
            let bytes = pe
                .data
                .get(offset..offset.checked_add(2).ok_or("header offset overflow")?)
                .ok_or("truncated PE header")?;
            Ok(u16::from_le_bytes(bytes.try_into().unwrap()))
        };
        let read32 = |offset: usize| -> Result<u32, String> {
            let bytes = pe
                .data
                .get(offset..offset.checked_add(4).ok_or("header offset overflow")?)
                .ok_or("truncated PE header")?;
            Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
        };
        let pe_offset = read32(0x3c)? as usize;
        if pe.data.get(0..2) != Some(b"MZ")
            || pe.data.get(pe_offset..pe_offset + 4) != Some(b"PE\0\0")
            || read16(pe_offset + 4)? != 0x8664
            || read16(pe.opt_header_offset)? != 0x20b
            || pe.opt_header_offset != pe_offset + 24
        {
            return Err("not an AMD64 PE32+ image".into());
        }
        let optional_size = read16(pe_offset + 20)? as usize;
        let optional_end = pe
            .opt_header_offset
            .checked_add(optional_size)
            .ok_or("optional header overflow")?;
        if optional_size < 112
            || optional_end > pe.size_of_headers as usize
            || optional_end > pe.data.len()
        {
            return Err("truncated declared PE32+ optional header".into());
        }
        let section_count = read16(pe_offset + 6)? as usize;
        if section_count != pe.sections.len()
            || optional_end != pe.section_table_offset
            || optional_end
                .checked_add(section_count * 40)
                .ok_or("section table overflow")?
                > pe.size_of_headers as usize
        {
            return Err("invalid declared section table extent".into());
        }
        let count = read32(pe.opt_header_offset + 108)? as usize;
        if count > (optional_size - 112) / 8 {
            return Err("data-directory count exceeds declared optional header".into());
        }
        if count <= 3 {
            return Ok(None);
        }
        let offset = pe.opt_header_offset + 112 + 3 * 8;
        let (rva, size) = (read32(offset)?, read32(offset + 4)?);
        match (rva, size) {
            (0, 0) => Ok(None),
            (0, _) | (_, 0) => Err("exception directory has only one of RVA/Size set".into()),
            _ => Ok(Some((rva, size))),
        }
    }

    fn read(&self, rva: u32, size: u32) -> Result<&'a [u8], String> {
        let end = rva.checked_add(size).ok_or("RVA extent overflow")?;
        if size == 0 || end > self.pe.size_of_image {
            return Err(format!("invalid mapped extent {rva:#x}+{size:#x}"));
        }
        if end <= self.pe.size_of_headers {
            return self
                .pe
                .data
                .get(rva as usize..end as usize)
                .ok_or_else(|| "truncated header mapping".into());
        }
        for section in &self.pe.sections {
            if rva >= section.virtual_address && end <= section.virtual_address + section.raw_size {
                let start =
                    u64::from(section.raw_address) + u64::from(rva - section.virtual_address);
                let end = start + u64::from(size);
                return self
                    .pe
                    .data
                    .get(start as usize..end as usize)
                    .ok_or_else(|| "truncated raw mapping".into());
            }
        }
        Err(format!(
            "extent {rva:#x}+{size:#x} is not wholly file-backed"
        ))
    }

    fn code_range(&self, begin: u32, end: u32) -> Result<(), String> {
        if begin == 0 || begin >= end {
            return Err("invalid runtime code interval".into());
        }
        self.read(begin, end - begin)?;
        if !self.pe.sections.iter().any(|s| {
            s.is_executable() && begin >= s.virtual_address && end <= s.virtual_address + s.raw_size
        }) {
            return Err("runtime range or handler is not executable".into());
        }
        Ok(())
    }
}

fn runtime_fields(bytes: &[u8]) -> (u32, u32, u32) {
    (
        u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
        u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
    )
}

fn resolve(map: &ImageMap<'_>, root: &RuntimeFunction) -> Coverage {
    match resolve_inner(map, root) {
        Ok(Some(handler)) => Coverage {
            kind: CoverageKind::Handler,
            handlers: vec![handler],
            issues: Vec::new(),
            runtime_ranges: Vec::new(),
        },
        Ok(None) => Coverage::unprotected(),
        Err(error) => {
            map.preserve_metadata_tail(root.unwind_data & !1);
            if let Some(rva) = map.metadata_cursor.get() {
                map.preserve_metadata_tail(rva);
            }
            Coverage::unknown(format!(
                "runtime {:#x}..{:#x}: {error}",
                root.begin_rva, root.end_rva
            ))
        }
    }
}

fn resolve_inner(
    map: &ImageMap<'_>,
    root: &RuntimeFunction,
) -> Result<Option<HandlerInfo>, String> {
    let mut current = (root.begin_rva, root.end_rva, root.unwind_data);
    let mut seen_unwind = HashSet::new();
    let mut seen_indirect = HashSet::new();
    let mut unwind_chain = Vec::new();
    let mut indirect_records = Vec::new();
    let mut chained_frame = None;
    // A resource bound also rejects unusually deep, acyclic adversarial chains.
    for _ in 0..256 {
        map.metadata_cursor.set(Some(current.2 & !1));
        map.code_range(current.0, current.1)?;
        let rva = current.2;
        if rva & 1 != 0 {
            let target = rva & !1;
            if target == 0 || target % 4 != 0 {
                return Err("misaligned indirect RUNTIME_FUNCTION RVA".into());
            }
            if !seen_indirect.insert(target) {
                return Err("indirect RUNTIME_FUNCTION cycle".into());
            }
            indirect_records.push(target);
            current = runtime_fields(map.read_metadata(target, 12)?);
            continue;
        }
        if rva == 0 || rva % 4 != 0 {
            return Err("zero or non-DWORD-aligned UNWIND_INFO RVA".into());
        }
        if !seen_unwind.insert(rva) {
            return Err("UNWIND_INFO chain cycle".into());
        }
        unwind_chain.push(rva);
        let header = map.read_metadata(rva, 4)?;
        let version = header[0] & 7;
        let flags = header[0] >> 3;
        if version != 1 {
            return Err(format!("unsupported UNWIND_INFO version {version}"));
        }
        if flags & !7 != 0 || flags & 4 != 0 && flags & 3 != 0 {
            return Err("invalid UNWIND_INFO flags".into());
        }
        if u32::from(header[1]) > current.1 - current.0 {
            return Err("prologue exceeds runtime interval".into());
        }
        if header[3] & 15 == 0 && header[3] >> 4 != 0 {
            return Err("frame offset without frame register".into());
        }
        if header[3] & 15 != 0 && !nonvolatile(header[3] & 15) {
            return Err("invalid nonvolatile frame register".into());
        }
        if chained_frame.is_some_and(|frame| frame != header[3]) {
            return Err("CHAININFO frame register/offset mismatch".into());
        }
        let padded_slots = (u32::from(header[2]) + 1) & !1;
        let length = 4 + padded_slots * 2;
        let bytes = map.read_metadata(rva, length)?;
        validate_codes(
            bytes,
            header[2] as usize,
            header[1],
            header[3],
            flags & 4 != 0,
        )?;
        // Padding occupies storage, but its value is not specified by the ABI.
        let tail = rva.checked_add(length).ok_or("unwind tail overflow")?;
        if flags & 4 != 0 {
            chained_frame = Some(header[3]);
            current = runtime_fields(map.read_metadata(tail, 12)?);
            continue;
        }
        if flags & 3 != 0 {
            let handler = u32::from_le_bytes(map.read_metadata(tail, 4)?.try_into().unwrap());
            let language_data_rva = tail.checked_add(4).ok_or("language data RVA overflow")?;
            map.preserve_metadata_tail(language_data_rva);
            map.code_range(
                handler,
                handler.checked_add(1).ok_or("handler RVA overflow")?,
            )?;
            return Ok(Some(HandlerInfo {
                runtime_begin_rva: root.begin_rva,
                runtime_end_rva: root.end_rva,
                unwind_rva: rva,
                handler_rva: handler,
                flags,
                language_data_rva,
                unwind_chain,
                indirect_records,
            }));
        }
        return Ok(None);
    }
    map.metadata_cursor.set(Some(current.2 & !1));
    Err("unwind/indirect chain exceeds inspection limit (256)".into())
}

fn nonvolatile(register: u8) -> bool {
    matches!(register, 3 | 5..=7 | 12..=15)
}

fn validate_codes(
    bytes: &[u8],
    count: usize,
    prologue: u8,
    frame: u8,
    chained: bool,
) -> Result<(), String> {
    let mut index = 0;
    let mut previous_offset = prologue;
    let mut frame_set = false;
    while index < count {
        let offset = bytes[4 + index * 2];
        let operation = bytes[5 + index * 2] & 15;
        let info = bytes[5 + index * 2] >> 4;
        if offset > previous_offset {
            return Err("UNWIND_CODE offsets exceed prologue or are not descending".into());
        }
        previous_offset = offset;
        let slots = match operation {
            0 if nonvolatile(info) => 1,
            1 if info == 0 => 2,
            1 if info == 1 => 3,
            2 => 1,
            3 if info == 0 && frame & 15 != 0 && !frame_set => {
                frame_set = true;
                1
            }
            4 if nonvolatile(info) => 2,
            5 if nonvolatile(info) => 3,
            8 if info >= 6 => 2,
            9 if info >= 6 => 3,
            10 if info <= 1 => 1,
            _ => {
                return Err(format!(
                    "unsupported/invalid unwind operation {operation}, OpInfo {info}"
                ));
            }
        };
        if index + slots > count {
            return Err("truncated multi-slot UNWIND_CODE".into());
        }
        if operation == 1 {
            let start = 6 + index * 2;
            let size = if info == 0 {
                u32::from(u16::from_le_bytes(
                    bytes[start..start + 2].try_into().unwrap(),
                )) * 8
            } else {
                u32::from_le_bytes(bytes[start..start + 4].try_into().unwrap())
            };
            if size == 0 || size % 8 != 0 {
                return Err("invalid UWOP_ALLOC_LARGE size".into());
            }
        }
        if operation == 5 || operation == 9 {
            let start = 6 + index * 2;
            let offset = u32::from_le_bytes(bytes[start..start + 4].try_into().unwrap());
            if offset % if operation == 5 { 8 } else { 16 } != 0 {
                return Err("misaligned far save offset".into());
            }
        }
        index += slots;
    }
    if frame & 15 != 0 && !frame_set && !chained {
        return Err("frame register has no UWOP_SET_FPREG".into());
    }
    Ok(())
}

#[cfg(test)]
#[path = "exception_tests.rs"]
mod tests;
