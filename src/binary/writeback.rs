//! Write recovered functions into the PE, then patch each entry trampoline to jmp to its recovered counterpart.

use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::binary::exception::{Coverage, CoverageKind, ExceptionTable};
use crate::binary::layout;
use crate::binary::manifest::{EntryOutcome, Manifest};
use crate::binary::pe::PeFile;
use crate::binary::pe_format as fmt;
use crate::binary::protection::SourceProtection;
use crate::codegen;
use crate::ir;
use crate::ir::regalloc;
use crate::ir::sched;
use crate::vm::discover::{self, VmEntry};
use crate::vm::explore;
use anyhow::{Context, Result, bail};

/// Per-function devirtualization results.
pub struct DevirtReport {
    /// Functions whose code was written and whose trampoline now points at it.
    pub retargeted: usize,
    /// Recovery did not finish
    pub incomplete: Vec<(u64, String)>,
    /// Entry patching or code generation was rejected, with the reason.
    pub failed: Vec<(u64, String)>,
    /// Image was signed and the signature has been removed.
    pub dropped_signature: bool,
    pub pdata_entries: usize,
    pub pdata_added: usize,
    pub checksum: u32,
    pub manifest: Manifest,
}

/// A swallowed entry whose original body is only INT3 padding.  There is no
/// room for the normal five-byte near jump, so the entry gets a two-byte short
/// jump to a five-byte E9 relay in a separately proven INT3 padding gap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RelayPatch {
    entry_va: u64,
    relay_va: u64,
}

impl DevirtReport {
    pub fn total(&self) -> usize {
        self.retargeted + self.incomplete.len() + self.failed.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RuntimeFunction {
    begin: u32,
    end: u32,
    unwind: u32,
}

impl RuntimeFunction {
    fn to_bytes(self) -> [u8; fmt::RUNTIME_FUNCTION_SIZE] {
        let mut b = [0u8; fmt::RUNTIME_FUNCTION_SIZE];
        b[0..4].copy_from_slice(&self.begin.to_le_bytes());
        b[4..8].copy_from_slice(&self.end.to_le_bytes());
        b[8..12].copy_from_slice(&self.unwind.to_le_bytes());
        b
    }
}

fn pe_checksum(data: &[u8], checksum_offset: usize) -> u32 {
    let mut sum: u32 = 0;
    let mut i = 0;
    while i + 1 < data.len() {
        let word = if i + 4 > checksum_offset && i < checksum_offset + 4 {
            0u32
        } else {
            u16::from_le_bytes([data[i], data[i + 1]]) as u32
        };
        sum += word;
        sum = (sum & 0xffff) + (sum >> 16);
        i += 2;
    }
    if data.len() % 2 == 1 {
        sum += data[data.len() - 1] as u32;
        sum = (sum & 0xffff) + (sum >> 16);
    }
    sum = (sum & 0xffff) + (sum >> 16);
    sum.wrapping_add(data.len() as u32)
}

/// Reject entries whose five-byte redirect would overwrite another entry or
/// cross a known function/section boundary. Discovery can identify swallowed
/// functions shorter than a JMP; identification alone is not permission to
/// overwrite their neighbours. Always compare against the original full scan,
/// including entries omitted by --only-entry or an incremental patch request.
fn patchable_entries(
    source: &PeFile,
    selected: &[VmEntry],
    vm_section: &str,
) -> (Vec<VmEntry>, Vec<(u64, String)>, Vec<RelayPatch>) {
    let all_entries = discover::find_vm_entries(source, vm_section);
    let functions = source.function_table();
    let padding = all_entries
        .iter()
        .filter(|e| e.int3_padding > 0)
        .map(|e| {
            (
                e.trampoline_va + 5,
                e.trampoline_va + 5 + e.int3_padding as u64,
            )
        })
        .collect::<Vec<_>>();
    let mut accepted = Vec::new();
    let mut rejected = Vec::new();
    let mut relays = Vec::new();
    let mut reserved_relays = Vec::<(u64, u64)>::new();
    let mut function_ends = HashMap::<u64, u64>::new();
    for &(begin, end) in &functions {
        function_ends
            .entry(begin)
            .and_modify(|stop| *stop = (*stop).min(end))
            .or_insert(end);
    }

    // Return the runtime-function range beginning at an entry.  Swallowed
    // aliases have a .pdata record but no E9 of their own; falling back to one
    // byte keeps the range proof conservative for malformed records.
    let function_range = |start: u64| -> (u64, u64) {
        (
            start,
            function_ends
                .get(&start)
                .copied()
                .unwrap_or(start.saturating_add(1)),
        )
    };
    let candidate_range = |entry: &VmEntry| -> (u64, u64) {
        let (begin, function_end) = function_range(entry.trampoline_va);
        let end = if source.read_u8(entry.trampoline_va) == Some(0xe9) {
            entry.trampoline_va.saturating_add(5)
        } else {
            function_end.max(begin.saturating_add(1))
        };
        (begin, end)
    };

    // A relay is valid only in an actual E9-following INT3 padding run.  It
    // must not overlap any .pdata function range or any discovered entry's
    // five-byte patch window, and the short jump must reach it.
    let find_relay = |entry: &VmEntry, reserved: &[(u64, u64)]| -> Option<u64> {
        let start = entry.trampoline_va;
        let patch_end = start.checked_add(2)?;
        let sec = source.section_for_va(start)?;
        if !sec.is_executable() {
            return None;
        }
        let reach_lo = patch_end.saturating_sub(128);
        let reach_hi = patch_end.saturating_add(127);
        for &(pad_lo, pad_hi) in &padding {
            let lo = pad_lo.max(reach_lo);
            // rel8 constrains the start, while all five bytes must stay in padding.
            let hi = pad_hi.min(reach_hi.saturating_add(5));
            if hi < lo.saturating_add(5) {
                continue;
            }
            let mut relay = lo;
            while relay + 5 <= hi {
                let rel = relay as i64 - patch_end as i64;
                if !(-128..=127).contains(&rel) {
                    relay += 1;
                    continue;
                }
                if source.read_va(relay, 5) != Some(&[0xcc; 5][..]) {
                    relay += 1;
                    continue;
                }
                if functions
                    .iter()
                    .any(|&(begin, end)| relay < end && begin < relay + 5)
                {
                    relay += 1;
                    continue;
                }
                if all_entries.iter().any(|other| {
                    let (lo, candidate_end) = candidate_range(other);
                    lo < relay + 5 && relay < candidate_end
                }) {
                    relay += 1;
                    continue;
                }
                if reserved
                    .iter()
                    .any(|&(lo, hi)| relay < hi && lo < relay + 5)
                {
                    relay += 1;
                    continue;
                }
                return Some(relay);
            }
        }
        None
    };

    for entry in selected {
        let start = entry.trampoline_va;
        let result = (|| {
            let Some(end) = start.checked_add(5) else {
                return Err("redirect address overflows".to_string());
            };
            let Some(rva) = source.va_to_rva(start) else {
                return Err("redirect is outside the image".to_string());
            };
            let Some(section) = source.section_for_rva(rva) else {
                return Err("redirect has no containing section".to_string());
            };
            let offset = (rva - section.virtual_address) as u64;
            if !section.is_executable()
                || offset + 5 > section.raw_size as u64
                || source.read_va(start, 5).is_none_or(|b| b.len() != 5)
            {
                return Err("redirect needs five file-backed executable bytes".to_string());
            }

            let (fn_begin, fn_stop) = function_range(start);
            let first = source.read_u8(start).unwrap_or(0);

            // A swallowed alias may use EB rel8 only when its own runtime
            // function supplies two INT3 bytes.  This deliberately excludes
            // one-byte .pdata entries: their second byte belongs to the next
            // padding gap and no nearby five-byte relay is proven by this
            // method.
            if first == 0xcc && (2..5).contains(&fn_stop.saturating_sub(fn_begin)) {
                if source.read_va(start, 2) != Some(&[0xcc; 2][..]) {
                    return Err("short relay requires two original INT3 bytes".to_string());
                }
                if functions.iter().any(|&(begin, stop)| {
                    (start < begin && begin < start + 2)
                        || (begin <= start && start < stop && stop < start + 2)
                }) || all_entries.iter().any(|other| {
                    other.trampoline_va != start && {
                        let (lo, hi) = candidate_range(other);
                        start < hi && lo < start + 2
                    }
                }) {
                    return Err(
                        "two-byte redirect overlaps a candidate or runtime boundary".to_string()
                    );
                }
                // Each entry owns its relay and generated body. Sharing by VM
                // address alone hides which body is actually reached and can
                // bypass another entry's exception preservation decision.
                let Some(relay) = find_relay(entry, &reserved_relays) else {
                    return Err("short relay has no reachable function-free INT3 padding".into());
                };
                reserved_relays.push((relay, relay + 5));
                let rel = relay as i64 - (start as i64 + 2);
                if !(-128..=127).contains(&rel) {
                    return Err("short relay is outside rel8 range".to_string());
                }
                return Ok(Some(RelayPatch {
                    entry_va: start,
                    relay_va: relay,
                }));
            }

            // A normal near jump is also safe for a swallowed record that has
            // at least five original INT3 bytes in its own function. This is
            // the ordinary E9 write path; EB is reserved for the true 2..4
            // byte windows above.
            let normal_e9_window = first == 0xe9
                || (first == 0xcc
                    && fn_stop.saturating_sub(fn_begin) >= 5
                    && source.read_va(start, 5) == Some(&[0xcc; 5][..]));
            if !normal_e9_window {
                return Err(format!(
                    "five-byte redirect crosses runtime function boundary {fn_begin:#x}..{fn_stop:#x}"
                ));
            }
            if let Some(&(begin, stop)) = functions.iter().find(|&&(begin, stop)| {
                (start < begin && begin < end) || (begin <= start && start < stop && stop < end)
            }) {
                return Err(format!(
                    "five-byte redirect crosses runtime function boundary {begin:#x}..{stop:#x}"
                ));
            }
            if all_entries.iter().any(|other| {
                other.trampoline_va != start && {
                    let (other_begin, other_end) = candidate_range(other);
                    start < other_end && other_begin < end
                }
            }) {
                return Err("five-byte redirect overlaps another candidate span".to_string());
            }
            Ok(None)
        })();
        match result {
            Ok(relay) => {
                accepted.push(*entry);
                if let Some(relay) = relay {
                    relays.push(relay);
                }
            }
            Err(reason) => rejected.push((start, reason)),
        }
    }
    (accepted, rejected, relays)
}

/// Compatibility wrapper: preserve source exception-dependent PCs by default.
pub fn write_devirt(
    pe: &PeFile,
    patch_base: Option<&PeFile>,
    entries: &[VmEntry],
    steps: usize,
    max_blocks: usize,
    timeout_secs: u64,
    vm_section: &str,
    jobs: usize,
) -> Result<(Vec<u8>, DevirtReport)> {
    write_devirt_with_policy(
        pe,
        patch_base,
        entries,
        steps,
        max_blocks,
        timeout_secs,
        vm_section,
        jobs,
        false,
    )
}

/// Fresh RX/R sections, original-entry CALLs, conservative exception screening.
/// Analysis-only mode records untranslated handler risks; it is not a runtime
/// replacement guarantee. Derived patch bases need a separately audited future
/// incremental writer and are deliberately refused here.
pub fn write_devirt_with_policy(
    pe: &PeFile,
    patch_base: Option<&PeFile>,
    selected: &[VmEntry],
    steps: usize,
    max_blocks: usize,
    timeout_secs: u64,
    vm_section: &str,
    jobs: usize,
    analysis_only: bool,
) -> Result<(Vec<u8>, DevirtReport)> {
    if patch_base.is_some_and(|base| base.data != pe.data) {
        bail!(
            "incremental patch bases are not supported by the preserve/manifest writer yet; rebuild from the original input"
        );
    }
    if jobs == 0 {
        bail!("worker count must be greater than zero");
    }
    let mut unique = std::collections::HashSet::new();
    if selected
        .iter()
        .any(|entry| !unique.insert(entry.trampoline_va))
    {
        bail!("duplicate selected trampoline address");
    }
    let table = Arc::new(ExceptionTable::parse(pe));
    if !table.global_errors.is_empty() {
        bail!(
            "invalid source exception directory: {}",
            table.global_errors.join("; ")
        );
    }
    let ck_off = pe.opt_header_offset + fmt::OPTIONAL_HEADER_CHECKSUM_OFFSET;
    let mut report = DevirtReport {
        retargeted: 0,
        incomplete: Vec::new(),
        failed: Vec::new(),
        dropped_signature: false,
        pdata_entries: table.records.len(),
        pdata_added: 0,
        checksum: u32::from_le_bytes(pe.data[ck_off..ck_off + 4].try_into().unwrap()),
        manifest: Manifest::new(pe, pe, analysis_only, None),
    };
    if analysis_only {
        report.manifest.limits.push("Analysis-only output may redirect PCs whose language handlers or unknown exception metadata were not translated; it is not a runtime replacement artifact.".into());
    }
    report.manifest.limits.push("Generated calls keep original entry addresses. Zero-emission output is byte-identical; nonidentical patch bases are refused.".into());
    let (patchable, failures, planned_relays) = patchable_entries(pe, selected, vm_section);
    for (entry, reason) in failures {
        record_failed(&mut report, entry, reason, Vec::new());
    }
    let mut entries = Vec::new();
    let mut initial_risks = Vec::new();
    for entry in patchable {
        let relay = planned_relays
            .iter()
            .find(|r| r.entry_va == entry.trampoline_va);
        let mut risks = Vec::new();
        let patch_size = if relay.is_some() { 2 } else { 5 };
        let end = entry
            .trampoline_va
            .checked_add(patch_size)
            .context("entry patch overflow")?;
        let mut metadata_overlap = table.patch_metadata_overlap(entry.trampoline_va, end);
        if let Some(relay) = relay {
            let relay_end = relay
                .relay_va
                .checked_add(5)
                .context("relay patch overflow")?;
            metadata_overlap = metadata_overlap
                .or_else(|| table.patch_metadata_overlap(relay.relay_va, relay_end));
        }
        // An analysis-only override permits an explicitly untransformed source
        // handler association. It never permits destroying original metadata
        // that another retained runtime record may still reference.
        if let Some((lo, hi)) = metadata_overlap {
            record_failed(
                &mut report,
                entry.trampoline_va,
                format!("entry/relay patch overlaps preserved exception metadata {lo:#x}..{hi:#x}"),
                Vec::new(),
            );
            continue;
        }
        add_coverage_risk(
            &mut risks,
            entry.trampoline_va,
            &table.coverage_range_va(entry.trampoline_va, end),
        );
        if let Some(relay) = relay {
            let end = relay
                .relay_va
                .checked_add(5)
                .context("relay patch overflow")?;
            add_coverage_risk(
                &mut risks,
                relay.relay_va,
                &table.coverage_range_va(relay.relay_va, end),
            );
        }
        if !analysis_only && !risks.is_empty() {
            record_failed(
                &mut report,
                entry.trampoline_va,
                "source entry/relay exception metadata requires preservation".into(),
                risks,
            );
        } else {
            entries.push(entry);
            initial_risks.push(risks);
        }
    }
    let relays: Vec<_> = planned_relays
        .into_iter()
        .filter(|r| entries.iter().any(|e| e.trampoline_va == r.entry_va))
        .collect();
    validate_relay_base(pe, &relays)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .context("building recovery pool")?;
    let total = entries.len();
    let width = total.max(1).to_string().len();
    let completed = AtomicUsize::new(0);
    let prepared: Vec<PreparedFn> = pool.install(|| entries.par_iter().zip(initial_risks.par_iter()).map(|(entry, risks)| {
        let started = std::time::Instant::now();
        let p = prepare_one(pe, entry, steps, max_blocks, timeout_secs, vm_section, table.clone(), analysis_only, risks);
        let n = completed.fetch_add(1, Ordering::Relaxed) + 1;
        let blocks = p.cfg.blocks.len(); let steps = p.cfg.total_steps; let elapsed = started.elapsed();
        match &p.incomplete {
            Some(reason) => eprintln!("  recover [{n:>width$}/{total}] {:#x}: incomplete ({reason}; {blocks} blocks, {steps} steps, {elapsed:.1?})", p.entry_va),
            None => eprintln!("  recover [{n:>width$}/{total}] {:#x}: ready ({blocks} blocks, {steps} steps, {elapsed:.1?})", p.entry_va),
        }
        p
    }).collect());
    let ready = prepared.iter().filter(|p| p.incomplete.is_none()).count();
    eprintln!(
        "recovery complete: {ready} ready, {} incomplete",
        prepared.len() - ready
    );
    if ready == 0 {
        for p in &prepared {
            record_preparation_failure(&mut report, p);
        }
        report.manifest.finish(&pe.data);
        return Ok((pe.data.clone(), report));
    }
    let code_rva = layout::next_code_rva(pe)?;
    let body_base = pe
        .image_base
        .checked_add(u64::from(code_rva))
        .context("code VA overflow")?;
    // Every body is emitted exactly once at the actual cursor. Original entry
    // addresses remain valid whether a sibling succeeds, fails, or is preserved.
    let redirects = codegen::CallRedirects::none();
    let mut offsets = Vec::new();
    let mut emitted = Vec::<Vec<u8>>::new();
    let mut unwind = Vec::<(u32, Vec<u8>)>::new();
    let mut cursor = 0u32;
    for p in &prepared {
        if p.incomplete.is_some() {
            record_preparation_failure(&mut report, p);
            offsets.push(None);
            emitted.push(Vec::new());
            continue;
        }
        let body_va = body_base
            .checked_add(u64::from(cursor))
            .context("body VA overflow")?;
        let patch_site = relays
            .iter()
            .find(|r| r.entry_va == p.entry_va)
            .map_or(p.entry_va, |r| r.relay_va);
        if let Err(error) = relative32(patch_site, body_va) {
            record_failed(
                &mut report,
                p.entry_va,
                error.to_string(),
                p.exception_risks.clone(),
            );
            offsets.push(None);
            emitted.push(Vec::new());
            continue;
        }
        let output = match codegen::emit_function_full(
            &p.cfg, &p.blocks, &p.allocs, body_va, pe, &redirects,
        ) {
            Ok(out) => out,
            Err(error) => {
                record_failed(
                    &mut report,
                    p.entry_va,
                    error.to_string(),
                    p.exception_risks.clone(),
                );
                offsets.push(None);
                emitted.push(Vec::new());
                continue;
            }
        };
        let len = u32::try_from(output.bytes.len()).context("body size exceeds32bits")?;
        if len == 0 {
            bail!("empty generated body for {:#x}", p.entry_va);
        }
        let next = cursor
            .checked_add(align_checked(len, 16)?)
            .context("code size overflow")?;
        code_rva.checked_add(next).context("code RVA overflow")?;
        offsets.push(Some(cursor));
        unwind.push((cursor, codegen::unwind_info(&output.prologue)));
        report.retargeted += 1;
        report.manifest.entry(EntryOutcome {
            entry_va: p.entry_va,
            status: "emitted".into(),
            reason: None,
            body_va: Some(body_va),
            body_size: Some(len),
            source_exception_risks: p.exception_risks.clone(),
        });
        eprintln!(
            "  emit   [{:>width$}/{ready}] {:#x} -> {body_va:#x}: {len} bytes",
            report.retargeted, p.entry_va
        );
        if std::env::var_os("TVM_MAP").is_some() {
            eprintln!("MAP {:#x} {body_va:#x} {len}", p.entry_va);
        }
        emitted.push(output.bytes);
        cursor = next;
    }
    if report.retargeted == 0 {
        report.manifest.finish(&pe.data);
        return Ok((pe.data.clone(), report));
    }
    let mut metadata = Vec::new();
    let mut unwind_offsets = Vec::new();
    for (code_off, blob) in &unwind {
        let off = u32::try_from(metadata.len()).context("unwind size exceeds32bits")?;
        unwind_offsets.push((*code_off, off));
        metadata.extend_from_slice(blob);
        metadata.resize(
            align_checked(u32::try_from(metadata.len())?, 4)? as usize,
            0,
        );
    }
    let runtime_offset = u32::try_from(metadata.len())?;
    let runtime_count = table
        .records
        .len()
        .checked_add(unwind.len())
        .context("runtime count overflow")?;
    let runtime_size = runtime_count
        .checked_mul(fmt::RUNTIME_FUNCTION_SIZE)
        .context("runtime size overflow")?;
    let total_metadata = metadata
        .len()
        .checked_add(runtime_size)
        .context("metadata size overflow")?;
    let plan = layout::plan_sections(pe, cursor as usize, total_metadata)?;
    if plan.code.rva != code_rva {
        bail!("final code RVA disagrees with emission");
    }
    let mut records: Vec<_> = table
        .records
        .iter()
        .map(|r| RuntimeFunction {
            begin: r.begin_rva,
            end: r.end_rva,
            unwind: r.unwind_data,
        })
        .collect();
    for (code_off, uw_off) in &unwind_offsets {
        let idx = offsets
            .iter()
            .position(|o| *o == Some(*code_off))
            .context("unwind/body inventory mismatch")?;
        let begin = code_rva
            .checked_add(*code_off)
            .context("body begin overflow")?;
        let end = begin
            .checked_add(u32::try_from(emitted[idx].len())?)
            .context("body end overflow")?;
        records.push(RuntimeFunction {
            begin,
            end,
            unwind: plan
                .metadata
                .rva
                .checked_add(*uw_off)
                .context("unwind RVA overflow")?,
        });
    }
    records.sort_by_key(|r| r.begin);
    if records.windows(2).any(|r| r[0].end > r[1].begin) {
        bail!("generated runtime ranges overlap");
    }
    for record in &records {
        metadata.extend_from_slice(&record.to_bytes());
    }
    if metadata.len() != total_metadata {
        bail!("runtime metadata size changed after planning");
    }
    let mut code = vec![0xcc; cursor as usize];
    for (idx, bytes) in emitted.iter().enumerate() {
        let Some(off) = offsets[idx] else { continue };
        code[off as usize..off as usize + bytes.len()].copy_from_slice(bytes);
        report.manifest.record_range(
            "code",
            "active",
            code_rva + off,
            bytes,
            Some(entries[idx].trampoline_va),
        )?;
    }
    report
        .manifest
        .record_range("code-section", "active", plan.code.rva, &code, None)?;
    for (idx, (_, blob)) in unwind.iter().enumerate() {
        report.manifest.record_range(
            "unwind",
            "active",
            plan.metadata.rva + unwind_offsets[idx].1,
            blob,
            None,
        )?;
    }
    let exception_rva = plan
        .metadata
        .rva
        .checked_add(runtime_offset)
        .context("runtime table RVA overflow")?;
    report.manifest.record_range(
        "runtime-table",
        "active",
        exception_rva,
        &metadata[runtime_offset as usize..],
        None,
    )?;
    report.manifest.record_range(
        "metadata-section",
        "active",
        plan.metadata.rva,
        &metadata,
        None,
    )?;
    if table.directory_size != 0 {
        let old = pe
            .read_va(
                pe.image_base + u64::from(table.directory_rva),
                table.directory_size as usize,
            )
            .context("original runtime table unreadable")?;
        report.manifest.record_range(
            "runtime-table",
            "superseded-preserved",
            table.directory_rva,
            old,
            None,
        )?;
    }
    let mut out = plan.write_sections(pe, &code, &metadata)?;
    let dir_count = u32::from_le_bytes(
        pe.data[pe.opt_header_offset + 108..pe.opt_header_offset + 112]
            .try_into()
            .unwrap(),
    );
    if dir_count <= fmt::IMAGE_DIRECTORY_ENTRY_SECURITY as u32 {
        bail!("optional header does not declare exception/security slots");
    }
    let security =
        fmt::data_directory_offset(pe.opt_header_offset, fmt::IMAGE_DIRECTORY_ENTRY_SECURITY);
    report.dropped_signature = out[security..security + 8].iter().any(|b| *b != 0);
    out[security..security + 8].fill(0);
    let exc =
        fmt::data_directory_offset(pe.opt_header_offset, fmt::IMAGE_DIRECTORY_ENTRY_EXCEPTION);
    write_u32_at(&mut out, exc, exception_rva);
    write_u32_at(&mut out, exc + 4, u32::try_from(runtime_size)?);
    report.pdata_entries = records.len();
    report.pdata_added = unwind.len();
    let mut written_relays = std::collections::HashSet::new();
    for relay in &relays {
        let idx = entries
            .iter()
            .position(|e| e.trampoline_va == relay.entry_va)
            .context("relay entry absent")?;
        if offsets[idx].is_none() || !written_relays.insert(relay.relay_va) {
            continue;
        }
        let (canonical, target) =
            emitted_relay_target(&entries, &relays, &offsets, relay.relay_va, body_base)?
                .context("relay has no emitted target")?;
        let raw = original_patch_offset(pe, relay.relay_va, 5)?;
        out[raw] = 0xe9;
        write_u32_at(
            &mut out,
            raw + 1,
            relative32(relay.relay_va, target)? as u32,
        );
        report.manifest.record_range(
            "relay",
            "active",
            pe.va_to_rva(relay.relay_va)
                .context("relay outside image")?,
            &out[raw..raw + 5],
            Some(canonical),
        )?;
        if std::env::var_os("TVM_MAP").is_some() {
            eprintln!(
                "RELAY {:#x} {:#x} {canonical:#x} {target:#x}",
                relay.entry_va, relay.relay_va
            );
        }
    }
    for (idx, entry) in entries.iter().enumerate() {
        let Some(off) = offsets[idx] else { continue };
        let va = entry.trampoline_va;
        let size = if let Some(relay) = relays.iter().find(|r| r.entry_va == va) {
            let rel = i8::try_from(i128::from(relay.relay_va) - (i128::from(va) + 2))
                .context("entry cannot reach relay")?;
            let raw = original_patch_offset(pe, va, 2)?;
            out[raw..raw + 2].copy_from_slice(&[0xeb, rel as u8]);
            2
        } else {
            let raw = original_patch_offset(pe, va, 5)?;
            let target = body_base
                .checked_add(u64::from(off))
                .context("body VA overflow")?;
            out[raw] = 0xe9;
            write_u32_at(&mut out, raw + 1, relative32(va, target)? as u32);
            5
        };
        let raw = original_patch_offset(pe, va, size)?;
        report.manifest.record_range(
            "entry-redirect",
            "active",
            pe.va_to_rva(va).context("entry outside image")?,
            &out[raw..raw + size],
            Some(va),
        )?;
    }
    write_u32_at(&mut out, ck_off, 0);
    report.checksum = pe_checksum(&out, ck_off);
    write_u32_at(&mut out, ck_off, report.checksum);
    verify_preserved_metadata(pe, &out, &table)?;
    report.manifest.finish(&out);
    Ok((out, report))
}

/// All source metadata still referenced by the copied runtime records keeps
/// its original mapping and bytes. This also catches an unsupported unwind
/// blob in apparently unused PE header space before an output can escape.
fn verify_preserved_metadata(pe: &PeFile, out: &[u8], table: &ExceptionTable) -> Result<()> {
    let headers_end = pe
        .section_table_offset
        .checked_add(
            pe.sections
                .len()
                .checked_mul(40)
                .context("source section-table extent overflow")?,
        )
        .context("source section-table extent overflow")?;
    let before = pe
        .data
        .get(pe.section_table_offset..headers_end)
        .context("source section headers unavailable")?;
    let after = out
        .get(pe.section_table_offset..headers_end)
        .context("output section headers unavailable")?;
    if before != after {
        bail!("source section mapping changed while preserving exception metadata");
    }
    for &(begin, end) in &table.metadata_ranges {
        if begin >= end || end > pe.size_of_image {
            bail!("invalid preserved exception metadata range");
        }
        let mut cursor = begin;
        while cursor < end {
            let (raw, length, next) = if cursor < pe.size_of_headers {
                let stop = end.min(pe.size_of_headers);
                (cursor as usize, (stop - cursor) as usize, stop)
            } else {
                let section = pe
                    .section_for_rva(cursor)
                    .context("preserved exception metadata lacks original mapping")?;
                let raw_end = section
                    .virtual_address
                    .checked_add(section.raw_size)
                    .context("metadata raw extent overflow")?;
                let mapped_end = section
                    .virtual_address
                    .checked_add(section.virtual_size.max(section.raw_size))
                    .context("metadata mapped extent overflow")?;
                let stop = end.min(mapped_end);
                let length = raw_end.min(stop).saturating_sub(cursor) as usize;
                // A virtual zero-fill tail has no file bytes. Its unchanged
                // section header above proves its extent and mapping remain.
                let raw = if length == 0 {
                    0
                } else {
                    section
                        .raw_address
                        .checked_add(cursor - section.virtual_address)
                        .context("metadata file offset overflow")? as usize
                };
                (raw, length, stop)
            };
            if next <= cursor {
                bail!("preserved metadata mapping made no progress");
            }
            if length != 0 {
                let raw_end = raw
                    .checked_add(length)
                    .context("metadata file range overflow")?;
                let original = pe
                    .data
                    .get(raw..raw_end)
                    .context("original exception metadata is truncated")?;
                let written = out
                    .get(raw..raw_end)
                    .context("preserved exception metadata is truncated")?;
                if original != written {
                    bail!("output changes preserved exception metadata {cursor:#x}..{next:#x}");
                }
            }
            cursor = next;
        }
    }
    Ok(())
}

fn add_coverage_risk(risks: &mut Vec<String>, pc: u64, coverage: &Coverage) {
    if coverage.kind == CoverageKind::Unprotected {
        return;
    }
    let message = format!(
        "source PC {pc:#x}: {:?}; handlers={:?}; issues={:?}; ranges={:?}",
        coverage.kind, coverage.handlers, coverage.issues, coverage.runtime_ranges
    );
    if !risks.contains(&message) {
        risks.push(message);
    }
}
fn record_failed(report: &mut DevirtReport, entry_va: u64, reason: String, risks: Vec<String>) {
    eprintln!("  patch {entry_va:#x}: rejected ({reason})");
    report.manifest.entry(EntryOutcome {
        entry_va,
        status: "preserved".into(),
        reason: Some(reason.clone()),
        body_va: None,
        body_size: None,
        source_exception_risks: risks,
    });
    report.failed.push((entry_va, reason));
}
fn record_preparation_failure(report: &mut DevirtReport, p: &PreparedFn) {
    let reason = p
        .incomplete
        .clone()
        .unwrap_or_else(|| "unavailable prepared graph".into());
    if p.exception_rejected {
        record_failed(report, p.entry_va, reason, p.exception_risks.clone());
    } else {
        report.manifest.entry(EntryOutcome {
            entry_va: p.entry_va,
            status: "incomplete".into(),
            reason: Some(reason.clone()),
            body_va: None,
            body_size: None,
            source_exception_risks: p.exception_risks.clone(),
        });
        report.incomplete.push((p.entry_va, reason));
    }
}
fn align_checked(value: u32, alignment: u32) -> Result<u32> {
    if !alignment.is_power_of_two() {
        bail!("invalid alignment");
    }
    value
        .checked_add(alignment - 1)
        .map(|n| n & !(alignment - 1))
        .context("aligned size overflow")
}
fn relative32(site: u64, target: u64) -> Result<i32> {
    let end = site
        .checked_add(5)
        .context("near branch address overflow")?;
    i32::try_from(i128::from(target) - i128::from(end))
        .context("entry/relay cannot reach emitted body with rel32")
}
fn original_patch_offset(pe: &PeFile, va: u64, size: usize) -> Result<usize> {
    let rva = pe.va_to_rva(va).context("patch outside image")?;
    let sec = pe.section_for_rva(rva).context("patch has no section")?;
    let offset = rva - sec.virtual_address;
    let size = u32::try_from(size)?;
    if !sec.is_executable()
        || offset
            .checked_add(size)
            .is_none_or(|end| end > sec.raw_size)
    {
        bail!("patch crosses executable file-backed range");
    }
    let raw = sec
        .raw_address
        .checked_add(offset)
        .context("patch raw offset overflow")? as usize;
    pe.data
        .get(
            raw..raw
                .checked_add(size as usize)
                .context("patch extent overflow")?,
        )
        .context("patch raw bytes unavailable")?;
    Ok(raw)
}

/// Pick a relay destination only from output bodies that were actually
/// emitted. An earlier alias may be incomplete or fail code generation; its
/// missing offset must never become a speculative branch destination.
fn emitted_relay_target(
    entries: &[VmEntry],
    relays: &[RelayPatch],
    offsets: &[Option<u32>],
    relay_va: u64,
    body_base: u64,
) -> Result<Option<(u64, u64)>> {
    if entries.len() != offsets.len() {
        bail!("relay recovery and emitted-offset inventories disagree");
    }
    for member in relays.iter().filter(|r| r.relay_va == relay_va) {
        let idx = entries
            .iter()
            .position(|e| e.trampoline_va == member.entry_va)
            .with_context(|| {
                format!(
                    "relay entry {:#x} absent from recovery inventory",
                    member.entry_va
                )
            })?;
        if let Some(offset) = offsets[idx] {
            let target = body_base
                .checked_add(offset as u64)
                .context("emitted relay target address overflows")?;
            return Ok(Some((member.entry_va, target)));
        }
    }
    Ok(None)
}

/// This planner has no persistent ownership map for earlier relays. Therefore
/// incremental relay sites must still be original INT3 bytes in the base.
/// Even an E9 to a derived body might belong to an unselected alias; rejecting
/// it avoids changing that alias's behavior by repurposing shared padding.
fn validate_relay_base(base: &PeFile, relays: &[RelayPatch]) -> Result<()> {
    for relay in relays {
        let entry = base.read_va(relay.entry_va, 2).with_context(|| {
            format!(
                "relay entry {:#x} is unreadable in patch base",
                relay.entry_va
            )
        })?;
        if entry != [0xcc, 0xcc] {
            bail!(
                "patch base short entry {:#x} is already occupied",
                relay.entry_va
            );
        }
        let body = base
            .read_va(relay.relay_va, 5)
            .with_context(|| format!("relay {:#x} is unreadable in patch base", relay.relay_va))?;
        if body != [0xcc, 0xcc, 0xcc, 0xcc, 0xcc] {
            bail!("patch base relay {:#x} is already occupied", relay.relay_va);
        }
    }
    Ok(())
}

/// Recover + schedule + allocate one function, returning reusable parts.
struct PreparedFn {
    entry_va: u64,
    incomplete: Option<String>,
    exception_risks: Vec<String>,
    exception_rejected: bool,
    cfg: ir::Cfg,
    blocks: Vec<sched::SchedBlock>,
    allocs: Vec<regalloc::Alloc>,
}

fn prepare_one(
    pe: &PeFile,
    entry: &VmEntry,
    steps: usize,
    max_blocks: usize,
    timeout_secs: u64,
    vm_section: &str,
    table: Arc<ExceptionTable>,
    analysis_only: bool,
    initial_risks: &[String],
) -> PreparedFn {
    let audit = SourceProtection::new(table, !analysis_only);
    let mut ex = explore::Explorer::with_vm_section(pe, crate::vm::DEFAULT_STACK_BASE, vm_section);
    ex.step_budget = steps;
    ex.block_budget = max_blocks;
    ex.time_budget = std::time::Duration::from_secs(timeout_secs);
    ex.source_protection = Some(audit.clone());

    let cfg = ex.recover(entry.vm_entry_va);
    let mut risks = initial_risks.to_vec();
    for (pc, coverage) in audit.risks() {
        add_coverage_risk(&mut risks, pc, &coverage);
    }
    if !analysis_only && !risks.is_empty() {
        return PreparedFn {
            entry_va: entry.trampoline_va,
            incomplete: Some("interpreted source PC has handler or unknown exception metadata; preserve original entry".into()),
            exception_risks: risks,
            exception_rejected: true,
            cfg,
            blocks: Vec::new(),
            allocs: Vec::new(),
        };
    }
    let mut prepared = prepare_recovered(entry.trampoline_va, cfg);
    prepared.exception_risks = risks;
    prepared
}

fn prepare_recovered(entry_va: u64, cfg: ir::Cfg) -> PreparedFn {
    // A rejected graph will never reach emission. Keep its diagnostic and
    // recovery statistics, but avoid building and allocating potentially huge
    // schedules for it (and never ask scheduling to interpret invalid SSA).
    if let Some(reason) = why_incomplete(&cfg) {
        return PreparedFn {
            entry_va,
            incomplete: Some(reason),
            exception_risks: Vec::new(),
            exception_rejected: false,
            cfg,
            blocks: Vec::new(),
            allocs: Vec::new(),
        };
    }
    let blocks = sched::schedule(&cfg);
    let allocs: Vec<regalloc::Alloc> = blocks.iter().map(regalloc::allocate).collect();

    PreparedFn {
        entry_va,
        incomplete: None,
        exception_risks: Vec::new(),
        exception_rejected: false,
        cfg,
        blocks,
        allocs,
    }
}

/// Why a recovered graph cannot be retargeted, or `None` when it can.
pub(crate) fn why_incomplete(cfg: &ir::Cfg) -> Option<String> {
    if cfg.blocks.is_empty() {
        return Some("no blocks recovered".to_string());
    }
    if cfg.timed_out {
        return Some("recovery timed out".to_string());
    }
    if cfg.unresolved != 0 {
        return Some(format!("{} block(s) never resolved", cfg.unresolved));
    }
    for block in &cfg.blocks {
        for event in &block.events {
            if let crate::ir::lift::Event::Call {
                target: None,
                import_slot: None,
                site,
                ..
            } = event
            {
                return Some(format!("unknown call target at {site:#x}"));
            }
        }
    }
    let issues = explore::validate(cfg);
    if !issues.is_clean() {
        return Some(format!("CFG invariant: {}", issues.summary()));
    }
    None
}

fn write_u32_at(buf: &mut Vec<u8>, off: usize, val: u32) {
    buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
}

#[cfg(test)]
mod preparation_tests {
    use super::*;
    use crate::ir::expr::{Arena, BlockRef, Reg};
    use crate::ir::{Block, BlockId, Terminator};

    fn graph() -> ir::Cfg {
        let id = BlockId {
            handler: 0x1000,
            vip: None,
        };
        ir::Cfg {
            entry: id,
            arena: Arena::default(),
            blocks: vec![Block {
                id,
                block_ref: BlockRef(0),
                params: vec![],
                events: vec![],
                from_vm_context: false,
                exit_regs: vec![],
                terminator: Terminator::TailCall { target: 0x2000 },
                cost: 17,
                preds: vec![],
            }],
            total_steps: 17,
            timed_out: false,
            unresolved: 0,
            back_edges: vec![],
            trivial_phis_removed: 0,
            stack_base: crate::vm::DEFAULT_STACK_BASE,
            native_entry_proofs: Default::default(),
            vm_range: (0, 0),
        }
    }

    #[test]
    fn incomplete_graph_keeps_reason_without_scheduling_invalid_expressions() {
        let mut cfg = graph();
        cfg.unresolved = 1;
        // Scheduling this foreign reference would index outside cfg.arena.
        // An already-rejected graph's expression roots must never be visited.
        let foreign = Arena::default().init_reg(Reg::Rax);
        cfg.blocks[0].exit_regs.push((Reg::Rax, foreign));
        let prepared = prepare_recovered(0x1000, cfg);
        assert_eq!(
            prepared.incomplete.as_deref(),
            Some("1 block(s) never resolved")
        );
        assert_eq!(prepared.cfg.total_steps, 17);
        assert!(prepared.blocks.is_empty());
        assert!(prepared.allocs.is_empty());
    }

    #[test]
    fn invariant_rejection_does_not_create_schedules_or_allocations() {
        let mut cfg = graph();
        cfg.blocks[0].terminator = Terminator::Jump(BlockId {
            handler: 0x3000,
            vip: None,
        });
        let expected = why_incomplete(&cfg).expect("a dangling CFG edge must be rejected");
        let prepared = prepare_recovered(0x1000, cfg);
        assert_eq!(prepared.incomplete.as_deref(), Some(expected.as_str()));
        assert!(prepared.blocks.is_empty());
        assert!(prepared.allocs.is_empty());
    }

    #[test]
    fn complete_graph_is_still_scheduled_and_allocated() {
        let prepared = prepare_recovered(0x1000, graph());
        assert!(prepared.incomplete.is_none());
        assert_eq!(prepared.blocks.len(), 1);
        assert_eq!(prepared.allocs.len(), 1);
    }
}

#[cfg(test)]
mod patch_boundary_tests {
    use super::*;

    const BASE: u64 = 0x180000000;
    const TEXT: u64 = BASE + 0x1000;
    const VM: u64 = BASE + 0x3000;

    fn fixture() -> PeFile {
        let mut data = vec![0; 0xa00];
        data[..2].copy_from_slice(b"MZ");
        write_u32_at(&mut data, 0x3c, 0x80);
        data[0x80..0x84].copy_from_slice(b"PE\0\0");
        data[0x84..0x86].copy_from_slice(&0x8664u16.to_le_bytes());
        data[0x86..0x88].copy_from_slice(&3u16.to_le_bytes());
        data[0x94..0x96].copy_from_slice(&0xf0u16.to_le_bytes());
        data[0x96..0x98].copy_from_slice(&0x2022u16.to_le_bytes());
        let opt = 0x98;
        data[opt..opt + 2].copy_from_slice(&0x20bu16.to_le_bytes());
        write_u32_at(&mut data, opt + 16, 0x1000);
        data[opt + 24..opt + 32].copy_from_slice(&BASE.to_le_bytes());
        for (off, value) in [
            (32, 0x1000),
            (36, 0x200),
            (56, 0x4000),
            (60, 0x400),
            (108, 16),
        ] {
            write_u32_at(&mut data, opt + off, value);
        }
        for (i, (name, rva, raw, flags)) in [
            (b".text\0\0\0", 0x1000, 0x400, 0x60000020),
            (b".pdata\0\0", 0x2000, 0x600, 0x40000040),
            (b".tvm0\0\0\0", 0x3000, 0x800, 0x60000020),
        ]
        .into_iter()
        .enumerate()
        {
            let sec = 0x188 + i * 40;
            data[sec..sec + 8].copy_from_slice(name);
            for (off, value) in [(8, 0x200), (12, rva), (16, 0x200), (20, raw), (36, flags)] {
                write_u32_at(&mut data, sec + off, value);
            }
        }
        data[0x400..0x600].fill(0xcc);
        data[0x400] = 0xe9;
        write_u32_at(&mut data, 0x401, (VM - TEXT - 5) as u32);
        // A real CALL to the alias must keep its original entry address.
        data[0x800] = 0xe8;
        write_u32_at(
            &mut data,
            0x801,
            ((TEXT + 5) as i64 - (VM + 5) as i64) as u32,
        );
        data[0x805] = 0xc3;
        let records = [
            (0x1000, 0x1005),
            (0x1005, 0x1007),
            (0x1007, 0x1020),
            (0x1020, 0x1028),
        ];
        for (i, (begin, end)) in records.into_iter().enumerate() {
            let at = 0x600 + i * fmt::RUNTIME_FUNCTION_SIZE;
            write_u32_at(&mut data, at, begin);
            write_u32_at(&mut data, at + 4, end);
            write_u32_at(&mut data, at + 8, 0x2080);
        }
        data[0x680] = 1;
        let dir = fmt::data_directory_offset(opt, fmt::IMAGE_DIRECTORY_ENTRY_EXCEPTION);
        write_u32_at(&mut data, dir, 0x2000);
        write_u32_at(
            &mut data,
            dir + 4,
            (records.len() * fmt::RUNTIME_FUNCTION_SIZE) as u32,
        );
        PeFile::parse(data).unwrap()
    }

    #[test]
    fn selected_alias_is_checked_against_unselected_original_entries() {
        let pe = fixture();
        let all = discover::find_vm_entries(&pe, ".tvm0");
        assert_eq!(
            all.iter().map(|e| e.trampoline_va).collect::<Vec<_>>(),
            vec![TEXT, TEXT + 5, TEXT + 7, TEXT + 0x20]
        );
        for selected in [all[1], all[2]] {
            let (accepted, rejected, relays) = patchable_entries(&pe, &[selected], ".tvm0");
            assert_eq!(accepted.len(), 1);
            assert!(rejected.is_empty());
            if selected.trampoline_va == TEXT + 5 {
                assert_eq!(relays.len(), 1);
                assert_eq!(relays[0].entry_va, selected.trampoline_va);
            } else {
                assert!(relays.is_empty());
            }
        }
        let (accepted, rejected, relays) = patchable_entries(&pe, &[all[0], all[3]], ".tvm0");
        assert_eq!(accepted.len(), 2);
        assert!(rejected.is_empty());
        assert!(relays.is_empty());
    }

    #[test]
    fn short_runtime_function_and_raw_section_end_are_rejected() {
        let mut pe = fixture();
        // Remove the +7 alias, leaving the short +5 function with no nearby
        // discovered entry. Its own .pdata end still forbids a five-byte patch.
        let dir =
            fmt::data_directory_offset(pe.opt_header_offset, fmt::IMAGE_DIRECTORY_ENTRY_EXCEPTION);
        write_u32_at(&mut pe.data, dir + 4, 2 * fmt::RUNTIME_FUNCTION_SIZE as u32);
        pe.data[0x618..0x630].fill(0);
        let all = discover::find_vm_entries(&pe, ".tvm0");
        // A one-byte runtime function has no two-byte entry sequence for EB;
        // keep it rejected even though the following padding is INT3.
        write_u32_at(&mut pe.data, 0x610, 0x1006);
        let (accepted, rejected, _) = patchable_entries(&pe, &[all[1]], ".tvm0");
        assert!(accepted.is_empty());
        assert!(rejected[0].1.contains("five-byte redirect crosses"));
        let beyond = VmEntry {
            trampoline_va: TEXT + 0x1fc,
            vm_entry_va: VM,
            int3_padding: 0,
        };
        let (accepted, rejected, _) = patchable_entries(&pe, &[beyond], ".tvm0");
        assert!(accepted.is_empty());
        assert!(rejected[0].1.contains("five file-backed executable bytes"));
    }

    #[test]
    fn rejected_alias_is_not_written_or_used_as_call_redirect() {
        let mut pe = fixture();
        // Make the swallowed alias a one-byte runtime function.  It remains a
        // discovered VM entry, but there is no two-byte window for an EB
        // redirect, so the call from the recovered VM body must keep its
        // original target instead of following a speculative redirect.
        write_u32_at(&mut pe.data, 0x610, 0x1006);
        let all = discover::find_vm_entries(&pe, ".tvm0");
        assert_eq!(all[1].trampoline_va, TEXT + 5);
        let (out, report) = write_devirt(&pe, None, &all[..2], 100, 8, 5, ".tvm0", 1).unwrap();
        assert_eq!(report.retargeted, 1);
        assert_eq!(report.failed.len(), 1);
        assert_eq!(report.failed[0].0, TEXT + 5);
        // Parse the new code section, then prove the CALL kept its original entry.
        let reparsed = PeFile::parse(out.clone()).unwrap();
        let section = reparsed.section_by_name(".tvcode").unwrap();
        let mut decoder = iced_x86::Decoder::with_ip(
            64,
            reparsed.section_bytes(section),
            BASE + u64::from(section.virtual_address),
            iced_x86::DecoderOptions::NONE,
        );
        let mut call_target = None;
        while decoder.can_decode() {
            let inst = decoder.decode();
            if inst.is_invalid() {
                break;
            }
            if inst.flow_control() == iced_x86::FlowControl::Call
                && matches!(
                    inst.op0_kind(),
                    iced_x86::OpKind::NearBranch16
                        | iced_x86::OpKind::NearBranch32
                        | iced_x86::OpKind::NearBranch64
                )
            {
                call_target = Some(inst.near_branch64());
                break;
            }
        }
        assert_eq!(call_target, Some(TEXT + 5));
        // Only the real E9 trampoline was patched; the rejected one is still
        // the original INT3 byte sequence.
        assert_eq!(&out[0x405..0x407], &[0xcc, 0xcc]);
    }

    #[test]
    fn each_short_alias_has_a_distinct_relay_and_actual_manifest_body() {
        let mut pe = fixture();
        write_u32_at(&mut pe.data, 0x61c, 0x1009);
        let all = discover::find_vm_entries(&pe, ".tvm0");
        let selected = [all[1], all[2]];
        let (_, _, relays) = patchable_entries(&pe, &selected, ".tvm0");
        assert_eq!(relays.len(), 2);
        assert_ne!(relays[0].relay_va, relays[1].relay_va);
        let (out, report) = write_devirt(&pe, None, &selected, 100, 8, 5, ".tvm0", 1).unwrap();
        assert_eq!(report.retargeted, 2);
        assert!(report.failed.is_empty());
        let result = PeFile::parse(out).unwrap();
        for entry in &report.manifest.entries {
            let short = result.read_va(entry.entry_va, 2).unwrap();
            assert_eq!(short[0], 0xeb);
            let relay_va = (entry.entry_va as i64 + 2 + i64::from(short[1] as i8)) as u64;
            let near = result.read_va(relay_va, 5).unwrap();
            assert_eq!(near[0], 0xe9);
            let body_va = (relay_va as i64
                + 5
                + i64::from(i32::from_le_bytes(near[1..].try_into().unwrap())))
                as u64;
            assert_eq!(Some(body_va), entry.body_va);
        }
        report.manifest.validate_output(&result).unwrap();
    }

    #[test]
    fn relay_target_never_borrows_another_alias_body() {
        let mut pe = fixture();
        write_u32_at(&mut pe.data, 0x61c, 0x1009);
        let all = discover::find_vm_entries(&pe, ".tvm0");
        let selected = [all[1], all[2]];
        let (_, _, relays) = patchable_entries(&pe, &selected, ".tvm0");
        assert_eq!(
            emitted_relay_target(
                &selected,
                &relays,
                &[None, Some(0x20)],
                relays[0].relay_va,
                BASE + 0x4000
            )
            .unwrap(),
            None
        );
        assert_eq!(
            emitted_relay_target(
                &selected,
                &relays,
                &[None, Some(0x20)],
                relays[1].relay_va,
                BASE + 0x4000
            )
            .unwrap(),
            Some((TEXT + 7, BASE + 0x4020))
        );
        assert!(
            emitted_relay_target(
                &selected[1..],
                &relays,
                &[Some(0)],
                relays[0].relay_va,
                BASE + 0x4000
            )
            .is_err()
        );
    }

    #[test]
    fn incomplete_alias_keeps_entry_and_relay_padding_untouched() {
        let pe = fixture();
        let all = discover::find_vm_entries(&pe, ".tvm0");
        let (_, _, relays) = patchable_entries(&pe, &all[1..2], ".tvm0");
        assert_eq!(relays.len(), 1);
        let (out, report) = write_devirt(&pe, None, &all[1..2], 0, 8, 5, ".tvm0", 1).unwrap();
        assert_eq!(report.retargeted, 0);
        assert_eq!(report.incomplete.len(), 1);
        assert_eq!(&out[0x400..0x600], &pe.data[0x400..0x600]);
    }

    #[test]
    fn unknown_unwind_in_relay_padding_is_preserved_in_both_policies() {
        let mut pe = fixture();
        // Another original runtime record points at the INT3 run chosen for
        // this alias's relay. Its unsupported header must survive unchanged,
        // even though the alias and relay PC themselves have no handler.
        write_u32_at(&mut pe.data, 0x608, 0x1028);
        let selected = discover::find_vm_entries(&pe, ".tvm0")
            .into_iter()
            .find(|entry| entry.trampoline_va == TEXT + 5)
            .unwrap();
        let (_, _, relays) = patchable_entries(&pe, &[selected], ".tvm0");
        assert_eq!(relays[0].relay_va, TEXT + 0x28);
        for analysis_only in [false, true] {
            let (out, report) = write_devirt_with_policy(
                &pe,
                None,
                &[selected],
                100,
                8,
                5,
                ".tvm0",
                1,
                analysis_only,
            )
            .unwrap();
            assert_eq!(out, pe.data);
            assert_eq!(report.retargeted, 0);
            assert_eq!(report.failed.len(), 1);
            assert!(
                report.failed[0]
                    .1
                    .contains("overlaps preserved exception metadata")
            );
        }
    }

    #[test]
    fn opaque_handler_data_container_preserves_an_otherwise_patchable_entry() {
        let mut pe = fixture();
        // Valid UNWIND_INFO in executable storage ends at +0x88. Its opaque
        // language-specific data can extend over the separate +0x100 entry;
        // without a format-specific extent proof, preserve this container.
        write_u32_at(&mut pe.data, 0x608, 0x1080);
        pe.data[0x480..0x484].copy_from_slice(&[9, 0, 0, 0]);
        write_u32_at(&mut pe.data, 0x484, 0x1000);
        pe.data[0x500] = 0xe9;
        write_u32_at(
            &mut pe.data,
            0x501,
            relative32(TEXT + 0x100, VM).unwrap() as u32,
        );
        for (offset, value) in [(0x630, 0x1100), (0x634, 0x1105), (0x638, 0x2080)] {
            write_u32_at(&mut pe.data, offset, value);
        }
        let dir =
            fmt::data_directory_offset(pe.opt_header_offset, fmt::IMAGE_DIRECTORY_ENTRY_EXCEPTION);
        write_u32_at(&mut pe.data, dir + 4, 5 * fmt::RUNTIME_FUNCTION_SIZE as u32);
        let selected = discover::find_vm_entries(&pe, ".tvm0")
            .into_iter()
            .find(|entry| entry.trampoline_va == TEXT + 0x100)
            .unwrap();
        let (patchable, failures, _) = patchable_entries(&pe, &[selected], ".tvm0");
        assert_eq!(patchable.len(), 1);
        assert!(failures.is_empty());
        assert_eq!(
            ExceptionTable::parse(&pe)
                .coverage_va(selected.trampoline_va)
                .kind,
            CoverageKind::Unprotected
        );
        for analysis_only in [false, true] {
            let (out, report) = write_devirt_with_policy(
                &pe,
                None,
                &[selected],
                100,
                8,
                5,
                ".tvm0",
                1,
                analysis_only,
            )
            .unwrap();
            assert_eq!(out, pe.data);
            assert_eq!(report.retargeted, 0);
            assert_eq!(report.failed.len(), 1);
            assert!(
                report.failed[0]
                    .1
                    .contains("overlaps preserved exception metadata")
            );
        }
    }

    #[test]
    fn source_unwind_aliasing_unused_section_headers_prevents_output() {
        let mut pe = fixture();
        // The fourth function has an unsupported all-zero UNWIND_INFO rooted
        // in the exact header slot the new code section would occupy.
        write_u32_at(&mut pe.data, 0x62c, 0x200);
        let selected = discover::find_vm_entries(&pe, ".tvm0")
            .into_iter()
            .find(|entry| entry.trampoline_va == TEXT)
            .unwrap();
        for analysis_only in [false, true] {
            let result = write_devirt_with_policy(
                &pe,
                None,
                &[selected],
                100,
                8,
                5,
                ".tvm0",
                1,
                analysis_only,
            );
            let error = match result {
                Ok(_) => panic!("metadata in a new section header was overwritten"),
                Err(error) => error,
            };
            assert!(
                error
                    .to_string()
                    .contains("output changes preserved exception metadata"),
                "{error}"
            );
        }
    }

    #[test]
    fn patch_base_relay_bytes_are_checked_before_reuse() {
        let mut base = fixture();
        let relay_va = TEXT + 0x28;
        let relay_off = 0x400 + 0x28;
        let entry_off = 0x405;
        let planned = [RelayPatch {
            entry_va: TEXT + 5,
            relay_va,
        }];
        // An untouched patch base is reusable. Any existing short jump or
        // relay bytes are treated as occupied because ownership of an earlier
        // relay is not persisted in the PE.
        assert!(validate_relay_base(&base, &planned).is_ok());
        let entry_rel = (relay_va as i64 - (TEXT + 5 + 2) as i64) as i8 as u8;
        base.data[entry_off..entry_off + 2].copy_from_slice(&[0xeb, entry_rel]);
        assert!(validate_relay_base(&base, &planned).is_err());
        base = fixture();
        base.data[relay_off] = 0xe9;
        write_u32_at(&mut base.data, relay_off + 1, 0x3000);
        assert!(validate_relay_base(&base, &planned).is_err());
        base = fixture();
        base.data[relay_off] = 0x90;
        assert!(validate_relay_base(&base, &planned).is_err());
    }
}
