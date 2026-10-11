//! Commands that emit one recovered function or write a patched binary.

use crate::cli::fmt::resolve_start;
use crate::{
    binary::pe, binary::writeback, codegen, ir, ir::regalloc, ir::sched, vm::discover, vm::explore,
};
use anyhow::{Context, Result, bail};
use iced_x86::{Decoder, DecoderOptions, Formatter, IntelFormatter};

fn select_jobs(cores: usize, jobs: Option<usize>, cpu_percent: Option<u8>) -> usize {
    const MAX_DEFAULT_JOBS: usize = 8;
    let cores = cores.max(1);
    if let Some(jobs) = jobs {
        // Preserve explicit --jobs behavior, including zero and oversubscription.
        jobs.max(1)
    } else if let Some(percent) = cpu_percent {
        // Widen before multiplying so even a large reported core count cannot overflow.
        ((cores as u128 * u128::from(percent) / 100) as usize).clamp(1, cores)
    } else {
        cores.min(MAX_DEFAULT_JOBS)
    }
}

pub fn cmd_write_devirt(
    input: &std::path::Path,
    output: &std::path::Path,
    steps: usize,
    max_blocks: usize,
    timeout: u64,
    vm_section: &str,
    jobs: Option<usize>,
    cpu_percent: Option<u8>,
    only_entry: &[u64],
    patch_base: Option<&std::path::Path>,
    analysis_only_ignore_eh: bool,
) -> Result<()> {
    let mut manifest_name = output.as_os_str().to_os_string();
    manifest_name.push(".layout.json");
    let manifest_path = std::path::PathBuf::from(manifest_name);
    let mut ida_name = output.as_os_str().to_os_string();
    ida_name.push(".ida.py");
    let ida_path = std::path::PathBuf::from(ida_name);
    for path in [output, manifest_path.as_path(), ida_path.as_path()] {
        if path.try_exists()? {
            bail!("output already exists; choose a new path: {}", path.display());
        }
    }
    let pe = pe::PeFile::load(input)?;
    let mut entries = discover::find_vm_entries(&pe, vm_section);
    if !only_entry.is_empty() {
        for va in only_entry {
            if !entries.iter().any(|e| e.trampoline_va == *va) {
                bail!("requested entry {va:#x} was not discovered in the original input");
            }
        }
        entries.retain(|e| only_entry.contains(&e.trampoline_va));
    }
    if entries.is_empty() {
        bail!("no VM entries selected");
    }
    let base = patch_base.map(pe::PeFile::load).transpose()?;
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let jobs = select_jobs(cores, jobs, cpu_percent);
    eprintln!("concurrency: {jobs} workers, {cores} available logical cores");
    if let Some(percent) = cpu_percent {
        eprintln!(
            "  --cpu-percent {percent}: concurrency budget, not a hard CPU utilization limit"
        );
    }
    eprintln!(
        "devirtualizing {} functions… ({jobs} at a time)",
        entries.len()
    );
    eprintln!("source exception policy: {}", if analysis_only_ignore_eh {
        "analysis-only; untranslated handlers may be lost"
    } else { "preserve original entries when source exception metadata needs translation" });
    let (out, report) = writeback::write_devirt_with_policy(
        &pe,
        base.as_ref(),
        &entries,
        steps,
        max_blocks,
        timeout,
        vm_section,
        jobs,
        analysis_only_ignore_eh,
    )?;
    let output_pe = pe::PeFile::parse(out.clone())?;
    report.manifest.validate_output(&output_pe)?;
    let manifest_bytes = serde_json::to_vec_pretty(&report.manifest)?;
    write_new_files(&[
        (&manifest_path, &manifest_bytes),
        (&ida_path, include_bytes!("../../scripts/prepare_ida.py")),
        (output, &out),
    ])?;
    println!("wrote {} bytes to {:?}", out.len(), output);
    println!("layout evidence: {}", manifest_path.display());
    println!("IDA entry/export visibility helper: {} (run in IDA after analysis)", ida_path.display());

    println!(
        "retargeted {}/{} functions",
        report.retargeted,
        report.total()
    );

    if !report.incomplete.is_empty() {
        println!("  {} failed to recover:", report.incomplete.len());
        let mut by_reason: std::collections::BTreeMap<&str, Vec<u64>> =
            std::collections::BTreeMap::new();
        for (va, reason) in &report.incomplete {
            by_reason.entry(reason.as_str()).or_default().push(*va);
        }
        for (reason, vas) in by_reason {
            println!("    {}: {reason}", fmt_vas(&vas));
        }
    }

    if !report.failed.is_empty() {
        println!(
            "  {} rejected during writeback or lowering:",
            report.failed.len()
        );
        let mut by_reason: std::collections::BTreeMap<&str, Vec<u64>> =
            std::collections::BTreeMap::new();
        for (va, reason) in &report.failed {
            by_reason.entry(reason.as_str()).or_default().push(*va);
        }
        for (reason, vas) in by_reason {
            println!("    {}: {reason}", fmt_vas(&vas));
        }
    }

    println!(
        "  unwind: {} RUNTIME_FUNCTION records ({} new)",
        report.pdata_entries, report.pdata_added
    );
    if report.pdata_added != report.retargeted {
        println!(
            "    WARNING: {} functions retargeted but got only {} unwind records",
            report.retargeted, report.pdata_added
        );
    }
    println!(
        "  checksum: {:#010x} sig: {}",
        report.checksum, report.dropped_signature
    );
    Ok(())
}

/// Create companions before the binary; never overwrite an existing path/alias.
/// On error, remove only files which this invocation successfully created.
fn write_new_files(files: &[(&std::path::Path, &[u8])]) -> Result<()> {
    use std::io::Write;
    let mut created = Vec::new();
    let result = (|| -> Result<()> {
        for &(path, bytes) in files {
            let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(path)
                .with_context(|| format!("creating {}", path.display()))?;
            created.push(path);
            file.write_all(bytes)?;
            file.sync_all()?;
        }
        Ok(())
    })();
    if result.is_err() {
        for path in created.into_iter().rev() { let _ = std::fs::remove_file(path); }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::{select_jobs, write_new_files};

    #[test]
    fn output_pair_never_overwrites_existing_binary_or_sidecar() {
        let dir = std::env::temp_dir().join(format!("tvm-output-pair-{}-{}", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir(&dir).unwrap();
        let output = dir.join("result.dll");
        let sidecar = dir.join("result.dll.layout.json");
        let script = dir.join("result.dll.ida.py");
        std::fs::write(&output, b"existing binary").unwrap();
        assert!(write_new_files(&[(&sidecar, b"report"), (&script, b"script"), (&output, b"replacement")]).is_err());
        assert_eq!(std::fs::read(&output).unwrap(), b"existing binary");
        assert!(!sidecar.exists());
        assert!(!script.exists());
        std::fs::write(&sidecar, b"existing report").unwrap();
        assert!(write_new_files(&[(&sidecar, b"report"), (&output, b"replacement")]).is_err());
        assert_eq!(std::fs::read(&sidecar).unwrap(), b"existing report");
        std::fs::remove_file(&output).unwrap();
        std::fs::remove_file(&sidecar).unwrap();
        std::fs::write(&script, b"existing helper").unwrap();
        assert!(write_new_files(&[(&sidecar, b"report"), (&script, b"script"), (&output, b"replacement")]).is_err());
        assert!(!sidecar.exists());
        assert!(!output.exists());
        assert_eq!(std::fs::read(&script).unwrap(), b"existing helper");
        std::fs::remove_file(&script).unwrap();
        write_new_files(&[(&sidecar, b"new report"), (&script, b"script"), (&output, b"new binary")]).unwrap();
        assert_eq!(std::fs::read(&output).unwrap(), b"new binary");
        assert_eq!(std::fs::read(&sidecar).unwrap(), b"new report");
        std::fs::remove_file(output).unwrap();
        std::fs::remove_file(sidecar).unwrap();
        std::fs::remove_file(script).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn cpu_percent_rounds_down_with_at_least_one_worker() {
        for (cores, percent, expected) in [
            (20, 60, 12),
            (3, 60, 1),
            (20, 1, 1),
            (1, 60, 1),
            (20, 100, 20),
            (1, 100, 1),
            (0, 60, 1),
            (usize::MAX, 100, usize::MAX),
        ] {
            assert_eq!(select_jobs(cores, None, Some(percent)), expected);
        }
    }

    #[test]
    fn default_jobs_preserves_eight_worker_cap() {
        for (cores, expected) in [(0, 1), (1, 1), (4, 4), (8, 8), (20, 8)] {
            assert_eq!(select_jobs(cores, None, None), expected);
        }
    }

    #[test]
    fn explicit_jobs_preserves_zero_and_oversubscription_behavior() {
        assert_eq!(select_jobs(20, Some(0), None), 1);
        assert_eq!(select_jobs(20, Some(2), None), 2);
        assert_eq!(select_jobs(20, Some(32), None), 32);
    }
}

/// Format a list of addresses, truncated so one bad pass cannot bury the summary.
fn fmt_vas(vas: &[u64]) -> String {
    const MAX: usize = 8;
    let shown: Vec<String> = vas.iter().take(MAX).map(|v| format!("{v:#x}")).collect();
    if vas.len() > MAX {
        format!("{} … and {} more", shown.join(" "), vas.len() - MAX)
    } else {
        shown.join(" ")
    }
}

/// Emit x86 machine code for one function and write it, or hex-dump to stdout.
#[allow(clippy::too_many_arguments)]
pub fn cmd_devirt(
    path: &std::path::Path,
    va: u64,
    output: Option<&std::path::Path>,
    stack_base: u64,
    steps: usize,
    max_blocks: usize,
    timeout: u64,
    dis: bool,
) -> Result<()> {
    let pe = pe::PeFile::load(path)?;
    let start = resolve_start(&pe, va);
    let mut ex = explore::Explorer::new(&pe, stack_base);
    ex.step_budget = steps;
    ex.block_budget = max_blocks;
    ex.time_budget = std::time::Duration::from_secs(timeout);
    let cfg = ex.recover(start);

    let unresolved = cfg
        .blocks
        .iter()
        .filter(|b| matches!(&b.terminator, ir::Terminator::Unresolved { .. }))
        .count();
    println!(
        "recovered {} blocks ({} unresolved) in {:.0?}",
        cfg.blocks.len(),
        unresolved,
        cfg.total_steps,
    );

    let bytes = emit_recovered(&cfg, start, &pe)?;

    println!("emitted {} bytes at base {:#x}", bytes.len(), start);
    if dis {
        let mut decoder = Decoder::with_ip(64, &bytes, start, DecoderOptions::NONE);
        let mut formatter = IntelFormatter::new();
        formatter.options_mut().set_hex_prefix("0x");
        formatter.options_mut().set_hex_suffix("");
        for instr in &mut decoder {
            let mut s = String::new();
            formatter.format(&instr, &mut s);
            println!("  {:#x}  {}", instr.ip(), s);
        }
        println!();
    }

    match output {
        Some(p) => {
            std::fs::write(p, &bytes)?;
            println!("wrote {:?}", p);
        }
        None => {
            // Hex dump — 16 bytes per line with address prefix.
            for (i, chunk) in bytes.chunks(16).enumerate() {
                let addr = start + (i * 16) as u64;
                let hex: Vec<String> = chunk.iter().map(|b| format!("{b:02x}")).collect();
                println!("{addr:#x}:  {}", hex.join(" "));
            }
        }
    }
    Ok(())
}

/// Apply the same recovery gate as batch writeback before touching SSA roots.
fn emit_recovered(cfg: &ir::Cfg, start: u64, pe: &pe::PeFile) -> Result<Vec<u8>> {
    if let Some(reason) = writeback::why_incomplete(cfg) {
        bail!("recovery is incomplete: {reason}");
    }
    let blocks = sched::schedule(cfg);
    let allocs: Vec<regalloc::Alloc> = blocks.iter().map(regalloc::allocate).collect();
    codegen::emit_function(cfg, &blocks, &allocs, start, pe)
}

#[cfg(test)]
mod recovery_gate_tests {
    use super::*;
    use crate::ir::expr::{Arena, BlockRef, Reg, Width};
    use crate::ir::{Block, BlockId, Cfg, Terminator};

    fn fixture() -> (Cfg, pe::PeFile) {
        let mut arena = Arena::default();
        let ret = arena.opaque("return_address", Width::W64);
        let first = BlockId { handler: 0x1000, vip: None };
        let second = BlockId { handler: 0x1010, vip: None };
        let cfg = Cfg {
            entry: first, arena,
            blocks: vec![
                Block { id: first, block_ref: BlockRef(0), params: vec![], events: vec![],
                    from_vm_context: false, exit_regs: vec![],
                    terminator: Terminator::Jump(second), cost: 1, preds: vec![] },
                Block { id: second, block_ref: BlockRef(1), params: vec![], events: vec![],
                    from_vm_context: false, exit_regs: vec![],
                    terminator: Terminator::Return { dest: ret }, cost: 1, preds: vec![first] },
            ],
            total_steps: 2, unresolved: 0, back_edges: vec![], trivial_phis_removed: 0,
            timed_out: false, stack_base: crate::vm::DEFAULT_STACK_BASE,
            native_entry_proofs: Default::default(), vm_range: (0x1000, 0x2000),
        };
        let pe = pe::PeFile {
            data: vec![], image_base: 0x1000, entry_point_rva: 0, size_of_image: 0x2000,
            sections: vec![], opt_header_offset: 0, section_table_offset: 0,
            file_alignment: 0x200, section_alignment: 0x1000,
            size_of_headers: 0, loader_bound: vec![],
        };
        (cfg, pe)
    }

    #[test]
    fn single_function_rejects_unidentified_vm_block_before_emission() {
        let (mut cfg, pe) = fixture();
        cfg.blocks[1].from_vm_context = true;
        let error = emit_recovered(&cfg, cfg.entry.handler, &pe).unwrap_err();
        assert!(error.to_string().contains("unidentified 1"), "{error}");
    }

    #[test]
    fn single_function_rejects_nondominating_parameter_before_emission() {
        let (mut cfg, pe) = fixture();
        // The first block reads a value defined only by its successor.
        let value = cfg.arena.param(BlockRef(1), Reg::Rax);
        cfg.blocks[0].exit_regs.push((Reg::Rax, value));
        let error = emit_recovered(&cfg, cfg.entry.handler, &pe).unwrap_err();
        assert!(error.to_string().contains("nondominating_params 1"), "{error}");
    }

    #[test]
    fn single_function_rejects_timeout_before_visiting_invalid_roots() {
        let (mut cfg, pe) = fixture();
        cfg.timed_out = true;
        let mut foreign = Arena::default();
        for n in 0..64 { foreign.constant(n, Width::W64); }
        cfg.blocks[0].exit_regs.push((Reg::Rax, foreign.init_reg(Reg::Rax)));
        let error = emit_recovered(&cfg, cfg.entry.handler, &pe).unwrap_err();
        assert!(error.to_string().contains("recovery timed out"), "{error}");
    }

    #[test]
    fn single_function_still_emits_complete_native_graph() {
        let (cfg, pe) = fixture();
        let bytes = emit_recovered(&cfg, cfg.entry.handler, &pe).unwrap();
        assert!(Decoder::with_ip(64, &bytes, cfg.entry.handler, DecoderOptions::NONE)
            .into_iter().any(|i| i.mnemonic() == iced_x86::Mnemonic::Ret));
    }
}
