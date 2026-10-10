use super::*;
use crate::binary::pe::Section;

const BASE: u64 = 0x180000000;
const TABLE: u64 = BASE + 0x1800;
const STACK: u64 = crate::vm::DEFAULT_STACK_BASE - 0x1000;

fn fixture() -> PeFile {
    PeFile { data: vec![0x90; 0x1200], image_base: BASE,
        entry_point_rva: 0x1000, size_of_image: 0x2000,
        sections: vec![Section { name: ".text".into(), virtual_address: 0x1000,
            virtual_size: 0x1000, raw_address: 0x200, raw_size: 0x1000,
            characteristics: 0x60000020 }], opt_header_offset: 0,
        section_table_offset: 0, file_alignment: 0x200, section_alignment: 0x1000,
        size_of_headers: 0x200, loader_bound: vec![] }
}

fn bounded_address(e: &mut Emulator<'_>, base:u64, bits:Width) -> Ref {
    let input=e.arena.init_reg(Reg::Rdx);
    let narrow=e.arena.trunc(input,bits);
    let offset=e.arena.zext(narrow,Width::W64);
    let base=e.arena.constant(base,Width::W64);
    e.arena.bin(BinOp::Add,base,offset)
}

fn historical_read_after_store(base:u64, bits:Width, target:u64) -> Option<u64> {
    let pe=fixture();
    let mut e=Emulator::new(&pe,crate::vm::DEFAULT_STACK_BASE);
    let pointer=e.arena.init_reg(Reg::Rcx);
    let old=e.load(pointer,Width::W32,BASE+0x1000);
    let address=bounded_address(&mut e,base,bits);
    let value=e.arena.constant(99,Width::W64);
    e.store(address,value,Width::W64,BASE+0x1008);
    e.pins.push((pointer,target));
    let rewritten=e.substitute(old);
    e.arena.as_const(rewritten)
}

#[test]
fn bounded_store_generation_disjoint_old_table_load_can_fold() {
    assert_eq!(historical_read_after_store(STACK,Width::W16,TABLE),Some(0x90909090));
    assert_eq!(historical_read_after_store(STACK,Width::W8,TABLE),Some(0x90909090));
}

#[test]
fn bounded_store_generation_wrapping_load_keeps_global_epoch() {
    let pe=fixture();
    let mut e=Emulator::new(&pe,crate::vm::DEFAULT_STACK_BASE);
    let pointer=e.arena.constant(u64::MAX-3,Width::W64);
    let old=e.arena.load(pointer,Width::W64);
    let address=bounded_address(&mut e,0,Width::W8);
    let value=e.arena.constant(99,Width::W64);
    e.store(address,value,Width::W64,BASE+0x1000);
    assert_ne!(e.arena.load(pointer,Width::W64),old);
}

#[test]
fn bounded_store_generation_signed_offset_is_not_an_unsigned_interval() {
    let pe=fixture();
    let mut e=Emulator::new(&pe,crate::vm::DEFAULT_STACK_BASE);
    let rdx=e.arena.init_reg(Reg::Rdx);
    let narrow=e.arena.trunc(rdx,Width::W16);
    let signed=e.arena.sext(narrow,Width::W64);
    let base=e.arena.constant(STACK,Width::W64);
    let address=e.arena.bin(BinOp::Add,base,signed);
    e.arena.bump_mem_gen();
    assert!(!e.arena.note_bounded_symbolic_store(address,8));
}

#[test]
fn bounded_store_generation_masked_offsets_include_all_possible_bytes() {
    let pe=fixture();
    for mask in [0x0f,0xff,0x1515,0xffff] {
        let mut e=Emulator::new(&pe,crate::vm::DEFAULT_STACK_BASE);
        let rdx=e.arena.init_reg(Reg::Rdx);
        let k=e.arena.constant(mask,Width::W64);
        let offset=e.arena.bin(BinOp::And,rdx,k);
        let base=e.arena.constant(STACK,Width::W64);
        let address=e.arena.bin(BinOp::Add,base,offset);
        e.arena.bump_mem_gen();
        assert!(e.arena.note_bounded_symbolic_store(address,8));
        for candidate in [STACK,STACK+mask,STACK+mask+7] {
            let p=e.arena.constant(candidate,Width::W64);
            assert!(!e.arena.load_version_is_current(p,Width::W8,0));
        }
        let outside=e.arena.constant(STACK+mask+32,Width::W64);
        assert!(e.arena.load_version_is_current(outside,Width::W8,0));
    }
}

#[test]
fn bounded_store_generation_does_not_use_narrow_shift_known_zero() {
    let pe=fixture();
    let mut e=Emulator::new(&pe,crate::vm::DEFAULT_STACK_BASE);
    let rdx=e.arena.init_reg(Reg::Rdx);
    let word=e.arena.trunc(rdx,Width::W32);
    let k=e.arena.constant(32,Width::W32);
    let shifted=e.arena.bin(BinOp::Shl,word,k);
    let offset=e.arena.zext(shifted,Width::W64);
    let base=e.arena.constant(STACK,Width::W64);
    let address=e.arena.bin(BinOp::Add,base,offset);
    e.arena.bump_mem_gen();
    assert!(!e.arena.note_bounded_symbolic_store(address,8));
}

#[test]
fn bounded_store_generation_clones_and_compaction_keep_full_interval() {
    let pe=fixture();
    let mut e=Emulator::new(&pe,crate::vm::DEFAULT_STACK_BASE);
    let address=bounded_address(&mut e,STACK,Width::W16);
    let value=e.arena.constant(99,Width::W64);
    e.store(address,value,Width::W64,BASE+0x1000);
    let mut clone=e.arena.clone();
    clone.compact(&[]);
    for candidate in [STACK,STACK+65535,STACK+65542] {
        let p=clone.constant(candidate,Width::W64);
        assert!(!clone.load_version_is_current(p,Width::W8,0));
    }
    let p=clone.constant(TABLE,Width::W64);
    assert!(clone.load_version_is_current(p,Width::W64,0));
}

#[test]
fn bounded_store_generation_unknown_boxed_extent_remains_global() {
    use iced_x86::code_asm::*;
    let mut pe=fixture();
    let mut a=CodeAssembler::new(64).unwrap();
    a.rep().stosq().unwrap();
    a.ret().unwrap();
    let code=a.assemble(BASE+0x1000).unwrap();
    pe.data[0x200..0x200+code.len()].copy_from_slice(&code);
    let mut e=Emulator::new(&pe,crate::vm::DEFAULT_STACK_BASE);
    let table=e.arena.constant(TABLE,Width::W64);
    let old=e.arena.load(table,Width::W64);
    assert!(matches!(e.run(BASE+0x1000,1),Stop::Budget{..}));
    assert_ne!(e.arena.load(table,Width::W64),old);
}

#[test]
fn bounded_store_generation_alias_and_width_boundary_remain_opaque() {
    for target in [TABLE,TABLE+255,TABLE+256,TABLE+262] {
        assert_eq!(historical_read_after_store(TABLE,Width::W8,target),None);
    }
}

#[test]
fn bounded_store_generation_wrap_and_unbounded_remain_global() {
    assert_eq!(historical_read_after_store(u64::MAX-3,Width::W8,TABLE),None);
    assert_eq!(historical_read_after_store(STACK,Width::W32,TABLE),None);
}

#[test]
fn bounded_store_generation_interning_preserves_only_disjoint_loads() {
    let pe=fixture();
    let mut e=Emulator::new(&pe,crate::vm::DEFAULT_STACK_BASE);
    let table=e.arena.constant(TABLE,Width::W64);
    let stack=e.arena.constant(STACK+128,Width::W64);
    let old_table=e.arena.load(table,Width::W64);
    let old_stack=e.arena.load(stack,Width::W64);
    let address=bounded_address(&mut e,STACK,Width::W16);
    let value=e.arena.constant(99,Width::W64);
    e.store(address,value,Width::W64,BASE+0x1008);
    assert_eq!(e.arena.load(table,Width::W64),old_table);
    assert_ne!(e.arena.load(stack,Width::W64),old_stack);
}

#[test]
fn bounded_store_generation_unknown_call_barrier_survives_later_bounded_write() {
    let pe=fixture();
    let mut e=Emulator::new(&pe,crate::vm::DEFAULT_STACK_BASE);
    let pointer=e.arena.init_reg(Reg::Rcx);
    let old=e.load(pointer,Width::W32,BASE+0x1000);
    e.clobber_call();
    let address=bounded_address(&mut e,STACK,Width::W16);
    let value=e.arena.constant(99,Width::W64);
    e.store(address,value,Width::W64,BASE+0x1008);
    e.pins.push((pointer,TABLE));
    let rewritten=e.substitute(old);
    assert!(matches!(e.arena.op(rewritten),Op::Load(..)));
}

#[test]
fn bounded_store_generation_real_x86_disjoint_write_preserves_table_read() {
    use iced_x86::code_asm::*;
    let mut pe=fixture();
    let mut a=CodeAssembler::new(64).unwrap();
    a.mov(eax,dword_ptr(rcx)).unwrap();
    a.movzx(edx,dx).unwrap();
    a.mov(rbp,STACK).unwrap();
    a.mov(qword_ptr(rbp+rdx),99).unwrap();
    a.ret().unwrap();
    let code=a.assemble(BASE+0x1000).unwrap();
    pe.data[0x200..0x200+code.len()].copy_from_slice(&code);
    let mut e=Emulator::new(&pe,crate::vm::DEFAULT_STACK_BASE);
    let pointer=e.state.reg(Reg::Rcx);
    assert!(matches!(e.run(BASE+0x1000,4),Stop::Budget{..}));
    let old=e.state.reg(Reg::Rax);
    e.pins.push((pointer,TABLE));
    let rewritten=e.substitute(old);
    assert_eq!(e.arena.as_const(rewritten),Some(0x90909090),
        "RDX={} / old={}",crate::ir::expr::render(&e.arena,e.state.reg(Reg::Rdx),8),crate::ir::expr::render(&e.arena,old,8));
}
