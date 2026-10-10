use super::*;
use crate::binary::pe::Section;
use crate::codegen;
use crate::ir::{regalloc, sched, Terminator};
use crate::vm::explore::Explorer;
use iced_x86::code_asm::*;

const BASE: u64 = 0x180000000;
const ENTRY: u64 = BASE + 0x1000;
const GLOBAL: u64 = BASE + 0x4000;
const STACK: u64 = crate::vm::DEFAULT_STACK_BASE;

fn fixture() -> PeFile {
    PeFile { data: vec![0x90;0x1200], image_base:BASE,
        entry_point_rva:0x1000,size_of_image:0x5000,
        sections:vec![
            Section{name:".text".into(),virtual_address:0x1000,virtual_size:0x1000,raw_address:0x200,raw_size:0x1000,characteristics:0x60000020},
            Section{name:".bss".into(),virtual_address:0x4000,virtual_size:0x1000,raw_address:0x1200,raw_size:0,characteristics:0xC0000080},
        ],opt_header_offset:0,section_table_offset:0,file_alignment:0x200,
        section_alignment:0x1000,size_of_headers:0x200,loader_bound:vec![] }
}

#[test]
fn mutable_bss_direct_read_remains_a_runtime_load() {
    let pe=fixture();let mut e=Emulator::new(&pe,STACK);
    let addr=e.arena.constant(GLOBAL,Width::W64);
    let value=e.load(addr,Width::W64,ENTRY);
    assert!(matches!(e.arena.op(value),Op::Load(..)));
    assert!(e.events.iter().any(|v| matches!(v,Event::Load{value:r,..} if *r==value)));
}

#[test]
fn mutable_bss_predicate_image_read_does_not_assume_loader_time() {
    let pe=fixture();let mut e=Emulator::new(&pe,STACK);
    assert!(e.image_load(GLOBAL,Width::W32).is_none());
}

#[test]
fn mutable_bss_pin_substitution_keeps_runtime_read() {
    let pe=fixture();let mut e=Emulator::new(&pe,STACK);
    let addr=e.arena.init_reg(Reg::Rcx);let old=e.load(addr,Width::W64,ENTRY);
    e.pins.push((addr,GLOBAL));let value=e.substitute(old);
    assert!(matches!(e.arena.op(value),Op::Load(..)));
}

#[test]
fn mutable_bss_readonly_zero_tail_can_still_fold() {
    let mut pe=fixture();pe.sections[1].characteristics=0x40000040;
    let mut e=Emulator::new(&pe,STACK);let addr=e.arena.constant(GLOBAL,Width::W64);
    let value=e.load(addr,Width::W64,ENTRY);assert_eq!(e.arena.as_const(value),Some(0));
}

#[test]
fn mutable_bss_abstract_stack_store_still_forwards() {
    let pe=fixture();let mut e=Emulator::new(&pe,STACK);
    let addr=e.arena.constant(STACK-0x80,Width::W64);let value=e.arena.constant(0x1234,Width::W64);
    e.store(addr,value,Width::W64,ENTRY);let result=e.load(addr,Width::W64,ENTRY+1);
    assert_eq!(e.arena.as_const(result),Some(0x1234));
}

#[test]
fn mutable_bss_actual_x86_retains_both_invocation_states() {
    let mut pe=fixture();let mut a=CodeAssembler::new(64).unwrap();let mut zero=a.create_label();
    a.mov(rax,qword_ptr(GLOBAL)).unwrap();a.test(rax,rax).unwrap();a.je(zero).unwrap();
    a.mov(eax,17).unwrap();a.ret().unwrap();a.set_label(&mut zero).unwrap();a.mov(eax,29).unwrap();a.ret().unwrap();
    let original=a.assemble(ENTRY).unwrap();pe.data[0x200..0x200+original.len()].copy_from_slice(&original);
    let cfg=Explorer::new(&pe,STACK).recover(ENTRY);
    assert!(cfg.blocks.iter().any(|b|matches!(b.terminator,Terminator::Branch{..})),"mutable global must retain both paths");
    assert_eq!(crate::binary::writeback::why_incomplete(&cfg),None);
    let blocks=sched::schedule(&cfg);let allocations:Vec<_>=blocks.iter().map(regalloc::allocate).collect();
    let generated=codegen::emit_function(&cfg,&blocks,&allocations,BASE+0x6000,&pe).unwrap();
    if let Some(out)=std::env::var_os("TVM_MUTABLE_BSS_FIXTURE_DIR") {
        let out=std::path::PathBuf::from(out);std::fs::create_dir_all(&out).unwrap();
        std::fs::write(out.join("original.bin"),original).unwrap();std::fs::write(out.join("generated.bin"),generated).unwrap();
    }
}
