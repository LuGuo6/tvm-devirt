#![allow(dead_code)]
// Independently exercised real PE/code bytes; no production helper is mocked.
#[path = "../src/binary/mod.rs"]
mod binary;
#[path = "../src/codegen/mod.rs"]
mod codegen;
#[path = "../src/ir/mod.rs"]
mod ir;
#[path = "../src/vm/mod.rs"]
mod vm;

use binary::pe::PeFile;
use binary::writeback::{write_devirt, write_devirt_with_policy};
use iced_x86::{Decoder, FlowControl};
const BASE: u64 = 0x180000000;
const TEXT: u64 = BASE + 0x1000;
const VM: u64 = BASE + 0x3000;
fn u16at(d: &mut [u8], o: usize, v: u16) {
    d[o..o + 2].copy_from_slice(&v.to_le_bytes());
}
fn u32at(d: &mut [u8], o: usize, v: u32) {
    d[o..o + 4].copy_from_slice(&v.to_le_bytes());
}
fn u64at(d: &mut [u8], o: usize, v: u64) {
    d[o..o + 8].copy_from_slice(&v.to_le_bytes());
}
fn rel(d: &mut [u8], o: usize, site: u64, target: u64, opcode: u8) {
    d[o] = opcode;
    u32at(d, o + 1, (target as i64 - (site + 5) as i64) as i32 as u32);
}
fn fixture(first: &[u8], second: &[u8], second_eh: bool) -> PeFile {
    let mut d = vec![0u8; 0xa00];
    d[..2].copy_from_slice(b"MZ");
    u32at(&mut d, 0x3c, 0x80);
    d[0x80..0x84].copy_from_slice(b"PE\0\0");
    u16at(&mut d, 0x84, 0x8664);
    u16at(&mut d, 0x86, 3);
    u16at(&mut d, 0x94, 0xf0);
    u16at(&mut d, 0x96, 0x2022);
    let o = 0x98;
    u16at(&mut d, o, 0x20b);
    u32at(&mut d, o + 16, 0x1000);
    u64at(&mut d, o + 24, BASE);
    u32at(&mut d, o + 32, 0x1000);
    u32at(&mut d, o + 36, 0x200);
    u32at(&mut d, o + 56, 0x4000);
    u32at(&mut d, o + 60, 0x400);
    u16at(&mut d, o + 68, 3);
    u32at(&mut d, o + 108, 16);
    u32at(&mut d, o + 112 + 3 * 8, 0x2000);
    u32at(&mut d, o + 116 + 3 * 8, 24);
    for (i, (name, rva, raw, chars)) in [
        (b".text\0\0\0", 0x1000, 0x400, 0x60000020u32),
        (b".pdata\0\0", 0x2000, 0x600, 0x40000040),
        (b".tvm0\0\0\0", 0x3000, 0x800, 0x60000020),
    ]
    .into_iter()
    .enumerate()
    {
        let s = 0x188 + i * 40;
        d[s..s + 8].copy_from_slice(name);
        u32at(&mut d, s + 8, 0x200);
        u32at(&mut d, s + 12, rva);
        u32at(&mut d, s + 16, 0x200);
        u32at(&mut d, s + 20, raw);
        u32at(&mut d, s + 36, chars);
    }
    d[0x400..0x600].fill(0xcc);
    rel(&mut d, 0x400, TEXT, VM, 0xe9);
    rel(&mut d, 0x420, TEXT + 0x20, VM + 0x40, 0xe9);
    d[0x500] = 0xc3;
    for (i, (begin, end, uw)) in [(0x1000, 0x1005, 0x2080), (0x1020, 0x1025, 0x2090)]
        .into_iter()
        .enumerate()
    {
        let p = 0x600 + i * 12;
        u32at(&mut d, p, begin);
        u32at(&mut d, p + 4, end);
        u32at(&mut d, p + 8, uw);
    }
    d[0x680] = 1;
    d[0x690] = if second_eh { 9 } else { 1 };
    u32at(&mut d, 0x694, 0x1100);
    d[0x800..0x800 + first.len()].copy_from_slice(first);
    d[0x840..0x840 + second.len()].copy_from_slice(second);
    PeFile::parse(d).unwrap()
}
fn selected(pe: &PeFile) -> Vec<vm::discover::VmEntry> {
    let all = vm::discover::find_vm_entries(pe, ".tvm0");
    [TEXT, TEXT + 0x20]
        .into_iter()
        .map(|va| *all.iter().find(|e| e.trampoline_va == va).unwrap())
        .collect()
}
fn body(pe: &PeFile, entry: u64) -> u64 {
    let b = pe.read_va(entry, 5).unwrap();
    assert_eq!(b[0], 0xe9);
    (entry as i64 + 5 + i32::from_le_bytes(b[1..5].try_into().unwrap()) as i64) as u64
}
fn code_to_section_end(pe: &PeFile, start: u64) -> &[u8] {
    let section = pe.section_for_va(start).unwrap();
    let end = pe.image_base + u64::from(section.virtual_address) + u64::from(section.virtual_size);
    pe.read_va(start, (end - start) as usize).unwrap()
}

#[test]
fn writer_no_emission_is_exact_input_identity() {
    let pe = fixture(&[0xc3], &[0xc3], false);
    let (out, report) = write_devirt(&pe, None, &selected(&pe), 0, 8, 5, ".tvm0", 1).unwrap();
    assert_eq!(report.retargeted, 0);
    assert!(
        out == pe.data,
        "a zero-emission run changed PE bytes: {} -> {} bytes",
        pe.data.len(),
        out.len()
    );
}

#[test]
fn writer_eh_rejection_keeps_original_entry_and_call_target() {
    let mut first = vec![0u8; 6];
    rel(&mut first, 0, VM, TEXT + 0x20, 0xe8);
    first[5] = 0xc3;
    let pe = fixture(&first, &[0xb8, 7, 0, 0, 0, 0xc3], true);
    let (out, report) = write_devirt(&pe, None, &selected(&pe), 100, 8, 5, ".tvm0", 1).unwrap();
    assert_eq!(
        report.retargeted, 1,
        "a native EH association may not be silently dropped"
    );
    assert_eq!(report.failed.len(), 1);
    assert_eq!(report.failed[0].0, TEXT + 0x20);
    let p = PeFile::parse(out).unwrap();
    assert_eq!(p.read_va(TEXT + 0x20, 5), pe.read_va(TEXT + 0x20, 5));
    let start = body(&p, TEXT);
    let calls: Vec<_> = Decoder::with_ip(64, code_to_section_end(&p, start), start, 0)
        .into_iter()
        .filter(|i| i.flow_control() == FlowControl::Call)
        .map(|i| i.near_branch64())
        .collect();
    assert!(
        calls.contains(&(TEXT + 0x20)),
        "call must reach the original EH-preserving entry; got {calls:x?}"
    );
}

#[test]
fn writer_call_keeps_original_entry_under_forward_layout() {
    let mut first = vec![0u8; 6];
    rel(&mut first, 0, VM, TEXT + 0x20, 0xe8);
    first[5] = 0xc3;
    let pe = fixture(&first, &[0xb8, 9, 0, 0, 0, 0xc3], false);
    let (out, report) = write_devirt(&pe, None, &selected(&pe), 100, 8, 5, ".tvm0", 1).unwrap();
    assert_eq!(report.retargeted, 2);
    assert!(report.failed.is_empty());
    let p = PeFile::parse(out).unwrap();
    let start = body(&p, TEXT);
    let jumps: Vec<_> = Decoder::with_ip(
        64,
        p.read_va(start, (body(&p, TEXT + 0x20) - start) as usize)
            .unwrap(),
        start,
        0,
    )
    .into_iter()
    .filter(|i| i.flow_control() == FlowControl::Call)
    .map(|i| i.near_branch64())
    .collect();
    assert!(
        jumps.contains(&(TEXT + 0x20)),
        "generated CALL must go through the original entry, not a speculative second-pass body: {jumps:x?}"
    );
    assert!(p.read_va(body(&p, TEXT + 0x20), 1).is_some());
}

#[test]
fn writer_counterexample_forward_redirect_shrinks_real_x86_tail_branch() {
    use ir::expr::{Arena, BlockRef, Reg};
    use ir::{Block, BlockId, Cfg, Terminator};
    let pe = fixture(&[0xc3], &[0xc3], false);
    let id = BlockId {
        handler: VM,
        vip: None,
    };
    let mut arena = Arena::new();
    let value = arena.constant(7, ir::expr::Width::W64);
    let cfg = Cfg {
        entry: id,
        arena,
        blocks: vec![Block {
            id,
            block_ref: BlockRef(0),
            params: vec![],
            events: vec![],
            from_vm_context: false,
            exit_regs: vec![(Reg::Rax, value)],
            terminator: Terminator::TailCall {
                target: TEXT + 0x20,
            },
            cost: 1,
            preds: vec![],
        }],
        total_steps: 1,
        unresolved: 0,
        back_edges: vec![],
        trivial_phis_removed: 0,
        timed_out: false,
        stack_base: vm::DEFAULT_STACK_BASE,
        native_entry_proofs: Default::default(),
        vm_range: (VM, VM + 0x200),
    };
    let blocks = ir::sched::schedule(&cfg);
    let allocs: Vec<_> = blocks.iter().map(ir::regalloc::allocate).collect();
    let start = BASE + 0x4000;
    let before = codegen::emit_function_full(
        &cfg,
        &blocks,
        &allocs,
        start,
        &pe,
        &codegen::CallRedirects::none(),
    )
    .unwrap();
    let mut redirects = codegen::CallRedirects::none();
    redirects.insert(TEXT + 0x20, start + 0x20);
    let after =
        codegen::emit_function_full(&cfg, &blocks, &allocs, start, &pe, &redirects).unwrap();
    let jb = Decoder::with_ip(64, &before.bytes, start, 0)
        .into_iter()
        .find(|i| i.flow_control() == FlowControl::UnconditionalBranch)
        .unwrap();
    let ja = Decoder::with_ip(64, &after.bytes, start, 0)
        .into_iter()
        .find(|i| i.flow_control() == FlowControl::UnconditionalBranch)
        .unwrap();
    assert_eq!(jb.len(), 5);
    assert_eq!(ja.len(), 2);
    assert_ne!(
        before.bytes.len(),
        after.bytes.len(),
        "the former writer's equal-length assumption is false"
    );
    eprintln!(
        "real tail branch: before={} bytes / {} aligned, after={} bytes / {} aligned",
        before.bytes.len(),
        (before.bytes.len() + 15) & !15,
        after.bytes.len(),
        (after.bytes.len() + 15) & !15
    );
}

#[test]
fn writer_nonidentical_incremental_base_is_explicitly_rejected() {
    let pe = fixture(&[0xc3], &[0xc3], false);
    let entries = selected(&pe);
    let (first, r1) = write_devirt(&pe, None, &entries[..1], 100, 8, 5, ".tvm0", 1).unwrap();
    assert_eq!(r1.retargeted, 1);
    let base = PeFile::parse(first).unwrap();
    let result = write_devirt(&pe, Some(&base), &entries[1..], 0, 8, 5, ".tvm0", 1);
    let error = result
        .err()
        .expect("nonidentical incremental bases must be explicitly refused");
    assert!(
        error.to_string().contains("incremental patch bases"),
        "{error:#}"
    );
}

#[test]
fn writer_preserves_long_overlay_bytes_at_existing_file_offsets() {
    let mut pe = fixture(&[0xc3], &[0xc3], false);
    let old_end = pe.data.len();
    pe.data
        .extend((0..0x6000).map(|i| ((i * 37 + 11) & 255) as u8));
    let selected = selected(&pe);
    let (out, report) = write_devirt(&pe, None, &selected[..1], 100, 8, 5, ".tvm0", 1).unwrap();
    assert_eq!(report.retargeted, 1);
    let mismatch = pe.data[old_end..]
        .iter()
        .zip(&out[old_end..pe.data.len()])
        .position(|(a, b)| a != b);
    assert!(
        mismatch.is_none(),
        "writer overwrote existing overlay at file offset {:?}",
        mismatch.map(|i| old_end + i)
    );
    let result = PeFile::parse(out).unwrap();
    let body_va = body(&result, TEXT);
    let code = result.section_for_va(body_va).unwrap();
    assert!(
        code.raw_address as usize >= pe.data.len(),
        "new payload must start after all original file bytes"
    );
}

#[test]
fn writer_preserves_old_executable_exception_table_and_separates_new_metadata() {
    let mut pe = fixture(&[0xc3], &[0xc3], false);
    let records = pe.data[0x600..0x618].to_vec();
    pe.data[0x900..0x918].copy_from_slice(&records);
    u32at(&mut pe.data, 0x98 + 112 + 3 * 8, 0x3100);
    let (out, report) =
        write_devirt(&pe, None, &selected(&pe)[..1], 100, 8, 5, ".tvm0", 1).unwrap();
    assert_eq!(report.retargeted, 1);
    let result = PeFile::parse(out).unwrap();
    assert_eq!(result.read_va(VM + 0x100, 24).unwrap(), records);
    assert_eq!(
        result.section_by_name(".tvm0").unwrap().characteristics,
        pe.section_by_name(".tvm0").unwrap().characteristics
    );
    let raw = &result.data;
    let dd = 0x98 + 112 + 3 * 8;
    let rva = u32::from_le_bytes(raw[dd..dd + 4].try_into().unwrap());
    let metadata = result.section_for_rva(rva).unwrap();
    assert!(
        !metadata.is_executable(),
        "new exception table is ordinary read-only data, not recovered code"
    );
    assert!(!metadata.is_writable());
}

#[test]
fn writer_preserves_eh_encountered_inside_recovered_instruction_path() {
    let mut pe = fixture(&[0x90, 0x90, 0xb8, 7, 0, 0, 0, 0xc3], &[0xc3], false);
    u32at(&mut pe.data, 0x98 + 116 + 3 * 8, 36);
    u32at(&mut pe.data, 0x618, 0x3002);
    u32at(&mut pe.data, 0x61c, 0x3008);
    u32at(&mut pe.data, 0x620, 0x20a0);
    pe.data[0x6a0] = 9;
    u32at(&mut pe.data, 0x6a4, 0x1100);
    let (out, report) =
        write_devirt(&pe, None, &selected(&pe)[..1], 100, 8, 5, ".tvm0", 1).unwrap();
    assert_eq!(
        report.retargeted, 0,
        "interior source PCs with handlers must veto replacement despite an unprotected entry"
    );
    assert!(
        out == pe.data,
        "a source EH veto without other emission must preserve input bytes"
    );
}

#[test]
fn writer_preserves_unknown_unwind_version_at_original_entry() {
    let mut pe = fixture(&[0xc3], &[0xc3], false);
    pe.data[0x680] = 3;
    let (out, report) =
        write_devirt(&pe, None, &selected(&pe)[..1], 100, 8, 5, ".tvm0", 1).unwrap();
    assert_eq!(
        report.retargeted, 0,
        "unknown version must not be interpreted as absence of native EH"
    );
    assert!(
        out == pe.data,
        "unrecognized EH metadata without other emission must preserve input bytes"
    );
}

#[test]
fn writer_short_alias_routes_match_distinct_manifest_bodies() {
    let mut pe = fixture(&[0xb8, 7, 0, 0, 0, 0xc3], &[0xc3], false);
    // All rows below use 0x2080. Clear the unused second unwind payload so the
    // legacy .pdata fallback scan cannot misread it as a range over the relays.
    pe.data[0x690..0x698].fill(0);
    u32at(&mut pe.data, 0x98 + 116 + 3 * 8, 48);
    for (i, (begin, end)) in [
        (0x1000, 0x1005),
        (0x1005, 0x1007),
        (0x1007, 0x1009),
        (0x1020, 0x1025),
    ]
    .into_iter()
    .enumerate()
    {
        u32at(&mut pe.data, 0x600 + i * 12, begin);
        u32at(&mut pe.data, 0x604 + i * 12, end);
        u32at(&mut pe.data, 0x608 + i * 12, 0x2080);
    }
    let entries: Vec<_> = vm::discover::find_vm_entries(&pe, ".tvm0")
        .into_iter()
        .filter(|e| [TEXT + 5, TEXT + 7].contains(&e.trampoline_va))
        .collect();
    assert_eq!(entries.len(), 2);
    let (out, report) = write_devirt(&pe, None, &entries, 100, 8, 5, ".tvm0", 1).unwrap();
    assert_eq!(report.retargeted, 2);
    assert!(report.failed.is_empty());
    let output = PeFile::parse(out).unwrap();
    let mut relays = Vec::new();
    let mut bodies = Vec::new();
    for entry in &report.manifest.entries {
        let bytes = output.read_va(entry.entry_va, 2).unwrap();
        assert_eq!(bytes[0], 0xeb);
        let relay = (entry.entry_va as i64 + 2 + i64::from(bytes[1] as i8)) as u64;
        assert_eq!(pe.read_va(relay, 5), Some(&[0xcc; 5][..]));
        let target = body(&output, relay);
        assert_eq!(Some(target), entry.body_va);
        relays.push(relay);
        bodies.push(target);
    }
    assert_ne!(relays[0], relays[1]);
    assert_ne!(bodies[0], bodies[1]);
    // Neither ordinary original trampoline was selected for replacement.
    assert_eq!(output.read_va(TEXT, 5), pe.read_va(TEXT, 5));
    assert_eq!(output.read_va(TEXT + 0x20, 5), pe.read_va(TEXT + 0x20, 5));
    report.manifest.validate_output(&output).unwrap();
}

#[test]
fn writer_analysis_only_reports_eh_risk_and_binds_actual_output() {
    use sha2::{Digest, Sha256};
    let pe = fixture(&[0xc3], &[0xb8, 7, 0, 0, 0, 0xc3], true);
    let entry = selected(&pe)[1];
    let (out, report) =
        write_devirt_with_policy(&pe, None, &[entry], 100, 8, 5, ".tvm0", 1, true).unwrap();
    assert_eq!(report.retargeted, 1);
    assert_eq!(report.manifest.exception_policy, "analysis-only");
    assert!(!report.manifest.whole_dll_runtime_verified);
    let row = &report.manifest.entries[0];
    assert!(!row.source_exception_risks.is_empty());
    assert_eq!(
        report.manifest.original_sha256,
        format!("{:X}", Sha256::digest(&pe.data))
    );
    assert_eq!(
        report.manifest.output_sha256,
        format!("{:X}", Sha256::digest(&out))
    );
    let output = PeFile::parse(out).unwrap();
    report.manifest.validate_output(&output).unwrap();
    let mut bad_range = report.manifest.clone();
    bad_range
        .ranges
        .iter_mut()
        .find(|range| range.kind == "code")
        .unwrap()
        .sha256 = "0".repeat(64);
    assert!(
        bad_range
            .validate_output(&output)
            .unwrap_err()
            .to_string()
            .contains("range hash")
    );
    let mut bad_binding = report.manifest.clone();
    bad_binding.entries[0].body_va = Some(row.body_va.unwrap() + 16);
    assert!(
        bad_binding
            .validate_output(&output)
            .unwrap_err()
            .to_string()
            .contains("active code range")
    );
    let mut bad_route = PeFile::parse(output.data.clone()).unwrap();
    // Keep whole-file and range hashes consistent to exercise route validation itself.
    bad_route.data[0x421] ^= 1;
    let mut matching_hashes = report.manifest.clone();
    matching_hashes.output_sha256 = format!("{:X}", Sha256::digest(&bad_route.data));
    matching_hashes
        .ranges
        .iter_mut()
        .find(|range| range.kind == "entry-redirect")
        .unwrap()
        .sha256 = format!("{:X}", Sha256::digest(&bad_route.data[0x420..0x425]));
    assert!(
        matching_hashes
            .validate_output(&bad_route)
            .unwrap_err()
            .to_string()
            .contains("entry route")
    );
    let mut tampered = PeFile::parse(output.data.clone()).unwrap();
    let rva = tampered.va_to_rva(row.body_va.unwrap()).unwrap();
    let section = tampered.section_for_rva(rva).unwrap();
    let raw = (section.raw_address + rva - section.virtual_address) as usize;
    tampered.data[raw] ^= 1;
    assert!(report.manifest.validate_output(&tampered).is_err());
}
