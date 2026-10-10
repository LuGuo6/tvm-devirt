//! Join regressions use real x86 bytes through Explorer, not hand-built SSA.
use super::*;
use crate::binary::pe::Section;
use iced_x86::code_asm::*;

const IMAGE: u64 = 0x180000000;
const ENTRY: u64 = IMAGE + 0x1000;
const ARM: u64 = IMAGE + 0x1100;
const JOIN: u64 = IMAGE + 0x1200;
const STACK: u64 = 0x7fff00003000;

fn id(handler: u64) -> BlockId { BlockId { handler, vip: None } }

fn image() -> PeFile {
    PeFile {
        data: vec![0xcc; 0x1200], image_base: IMAGE, entry_point_rva: 0x1000,
        size_of_image: 0x3000,
        sections: vec![Section { name: ".text".into(), virtual_address: 0x1000,
            virtual_size: 0x1000, raw_address: 0x200, raw_size: 0x1000,
            characteristics: 0x60000020 }],
        opt_header_offset: 0, section_table_offset: 0, file_alignment: 0x200,
        section_alignment: 0x1000, size_of_headers: 0x200, loader_bound: vec![],
    }
}

fn put(pe: &mut PeFile, address: u64, emit: impl FnOnce(&mut CodeAssembler)) {
    let mut a = CodeAssembler::new(64).unwrap();
    emit(&mut a);
    let bytes = a.assemble(address).unwrap();
    let offset = (address - ENTRY + 0x200) as usize;
    pe.data[offset..offset + bytes.len()].copy_from_slice(&bytes);
}

fn diamond() -> PeFile {
    let mut pe = image();
    put(&mut pe, ENTRY, |a| {
        a.test(ecx, ecx).unwrap();
        a.je(ARM).unwrap();
        a.mov(r10d, 11).unwrap();
        a.test(edx, edx).unwrap();
        a.je(JOIN).unwrap();
        a.mov(eax, -1).unwrap();
        a.ret().unwrap();
    });
    put(&mut pe, ARM, |a| {
        a.mov(r10d, 22).unwrap();
        a.test(edx, edx).unwrap();
        a.je(JOIN).unwrap();
        a.mov(eax, -2).unwrap();
        a.ret().unwrap();
    });
    put(&mut pe, JOIN, |a| { a.mov(eax, r10d).unwrap(); a.ret().unwrap(); });
    pe
}

fn export(kind: &str, pe: &PeFile, cfg: &Cfg) {
    use crate::codegen::{CallRedirects, emit_function_full};
    use crate::ir::{sched, regalloc};
    let blocks = sched::schedule(cfg);
    let allocs: Vec<_> = blocks.iter().map(regalloc::allocate).collect();
    let emitted = emit_function_full(cfg, &blocks, &allocs, IMAGE + 0x5000, pe, &CallRedirects::none()).unwrap();
    if let Some(dir) = std::env::var_os("TVM_NATIVE_JOIN_FIXTURE_DIR") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(format!("{kind}.original.bin")), &pe.data[0x200..]).unwrap();
        std::fs::write(dir.join(format!("{kind}.generated.bin")), emitted.bytes).unwrap();
    }
}

#[test]
fn native_join_diamond_carries_both_values_instead_of_first_predecessor() {
    let pe = diamond();
    let cfg = Explorer::new(&pe, STACK).recover(ENTRY);
    export("diamond", &pe, &cfg);
    assert_eq!(cfg.unresolved, 0);
    let join = cfg.block(id(JOIN)).unwrap();
    assert_eq!(join.preds.len(), 2);
    assert!(join.params.iter().any(|(r, _)| *r == Reg::R10),
        "native join must not freeze 11 or 22 from whichever arm arrived first");
}

#[test]
fn native_join_loop_carries_current_cursor() {
    let mut pe = image();
    put(&mut pe, ENTRY, |a| {
        a.mov(r10, rcx).unwrap();
        a.test(edx, edx).unwrap();
        a.je(JOIN).unwrap();
        a.mov(eax, -1).unwrap();
        a.ret().unwrap();
    });
    put(&mut pe, JOIN, |a| {
        a.movzx(eax, byte_ptr(r10)).unwrap();
        a.inc(r10).unwrap();
        a.test(eax, eax).unwrap();
        a.jne(JOIN).unwrap();
        a.mov(rax, r10).unwrap();
        a.ret().unwrap();
    });
    let cfg = Explorer::new(&pe, STACK).recover(ENTRY);
    export("cursor_loop", &pe, &cfg);
    assert_eq!(cfg.unresolved, 0);
    let join = cfg.block(id(JOIN)).unwrap();
    assert!(join.params.iter().any(|(r, _)| *r == Reg::R10),
        "backedge must carry its incremented cursor");
}

fn single_predecessor_preserves_load_before_later_store(in_vm: bool) {
    const GLOBAL: u64 = IMAGE + 0x2000;
    let mut pe = image();
    if in_vm { pe.sections[0].name = ".tvm0".into(); }
    pe.data.resize(0x2200, 0);
    pe.sections.push(Section { name: ".data".into(), virtual_address: 0x2000,
        virtual_size: 0x1000, raw_address: 0x1200, raw_size: 0x1000,
        characteristics: 0xc0000040 });
    put(&mut pe, ENTRY, |a| {
        a.mov(r11, GLOBAL).unwrap();
        a.mov(r10d, dword_ptr(r11)).unwrap();
        a.test(ecx, ecx).unwrap();
        a.jne(JOIN).unwrap();
        a.mov(eax, -1).unwrap();
        a.ret().unwrap();
    });
    put(&mut pe, JOIN, |a| {
        a.mov(dword_ptr(r11), edx).unwrap();
        a.mov(eax, r10d).unwrap();
        a.ret().unwrap();
    });
    let cfg = Explorer::new(&pe, STACK).recover(ENTRY);
    export(if in_vm { "vm_load_before_store" } else { "load_before_store" }, &pe, &cfg);
    assert_eq!(cfg.unresolved, 0);
    let join = cfg.block(id(JOIN)).unwrap();
    assert_eq!(join.preds.len(), 1);
    assert!(join.params.iter().any(|(r, _)| *r == Reg::R10),
        "one predecessor still needs its already-read load value, not a reload after the store");
}

#[test]
fn native_join_single_predecessor_preserves_load_before_later_store() {
    single_predecessor_preserves_load_before_later_store(false);
}

#[test]
fn native_join_vm_section_single_predecessor_preserves_old_load() {
    single_predecessor_preserves_load_before_later_store(true);
}

#[test]
fn native_join_vm_section_diamond_requires_complete_machine_history() {
    let mut pe = diamond();
    pe.sections[0].name = ".tvm0".into();
    let cfg = Explorer::new(&pe, STACK).recover(ENTRY);
    export("vm_diamond", &pe, &cfg);
    assert_eq!(cfg.unresolved, 0);
    let join = cfg.block(id(JOIN)).unwrap();
    assert!(join.params.iter().any(|(r, _)| *r == Reg::R10));
}

fn call_result_diamond(capture_branch_input: bool) -> PeFile {
    let mut pe = image();
    const HELPER_A: u64 = IMAGE + 0x1800;
    const HELPER_B: u64 = IMAGE + 0x1810;
    put(&mut pe, ENTRY, |a| {
        a.push(rbx).unwrap();
        a.sub(rsp, 0x20).unwrap();
        a.mov(ebx, edx).unwrap();
        a.test(ecx, ecx).unwrap();
        a.je(ARM).unwrap();
        a.call(HELPER_A).unwrap();
        a.mov(r10, rax).unwrap();
        a.xor(eax, eax).unwrap();
        if capture_branch_input { a.test(ebx, ebx).unwrap(); }
        else { a.test(edx, edx).unwrap(); }
        a.je(JOIN).unwrap();
        a.mov(eax, -1).unwrap();
        a.add(rsp, 0x20).unwrap();
        a.pop(rbx).unwrap();
        a.ret().unwrap();
    });
    put(&mut pe, ARM, |a| {
        a.call(HELPER_B).unwrap();
        a.mov(r10, rax).unwrap();
        a.xor(eax, eax).unwrap();
        if capture_branch_input { a.test(ebx, ebx).unwrap(); }
        else { a.test(edx, edx).unwrap(); }
        a.je(JOIN).unwrap();
        a.mov(eax, -2).unwrap();
        a.add(rsp, 0x20).unwrap();
        a.pop(rbx).unwrap();
        a.ret().unwrap();
    });
    put(&mut pe, JOIN, |a| {
        a.mov(eax, dword_ptr(r10)).unwrap();
        a.add(rsp, 0x20).unwrap();
        a.pop(rbx).unwrap();
        a.ret().unwrap();
    });
    put(&mut pe, HELPER_A, |a| { a.ret().unwrap(); });
    put(&mut pe, HELPER_B, |a| { a.ret().unwrap(); });
    pe
}

#[test]
fn native_join_call_result_diamond_seeds_exact_pointer_origins() {
    let pe = call_result_diamond(true);
    let cfg = Explorer::new(&pe, STACK).recover(ENTRY);
    assert_eq!(cfg.unresolved, 0, "exact call-return register origins must be parameterized");
    let join = cfg.block(id(JOIN)).unwrap();
    assert!(join.params.iter().any(|(r, _)| *r == Reg::R10));
    export("call_result_diamond", &pe, &cfg);
}

#[test]
fn native_join_call_diamond_rejects_uncaptured_volatile_branch_input() {
    // The call model does not capture EDX. Parameterizing it used to produce a
    // seemingly complete CFG that selected the wrong branch in real x86 runs.
    let pe = call_result_diamond(false);
    let cfg = Explorer::new(&pe, STACK).recover(ENTRY);
    assert!(cfg.unresolved > 0,
        "an uncaptured volatile call output must not obtain a native join proof");
}

#[test]
fn native_join_single_predecessor_does_not_parameterize_uncaptured_call_output() {
    let mut pe = image();
    const HELPER: u64 = IMAGE + 0x1800;
    put(&mut pe, ENTRY, |a| {
        a.push(rbx).unwrap();
        a.sub(rsp, 0x20).unwrap();
        a.mov(ebx, ecx).unwrap();
        a.call(HELPER).unwrap();
        a.test(ebx, ebx).unwrap();
        a.je(ARM).unwrap();
        a.mov(eax, edx).unwrap();
        a.add(rsp, 0x20).unwrap();
        a.pop(rbx).unwrap();
        a.ret().unwrap();
    });
    put(&mut pe, ARM, |a| {
        a.mov(eax, 0).unwrap();
        a.add(rsp, 0x20).unwrap();
        a.pop(rbx).unwrap();
        a.ret().unwrap();
    });
    put(&mut pe, HELPER, |a| { a.ret().unwrap(); });
    let cfg = Explorer::new(&pe, STACK).recover(ENTRY);
    let blocks = crate::ir::sched::schedule(&cfg);
    let allocs: Vec<_> = blocks.iter().map(crate::ir::regalloc::allocate).collect();
    let result = crate::codegen::emit_function_full(&cfg, &blocks, &allocs,
        IMAGE + 0x5000, &pe, &crate::codegen::CallRedirects::none());
    assert!(result.is_err(),
        "single-predecessor graft must not invent an architectural home for a call-clobbered value");
}
