use super::*;
use crate::binary::pe::Section;

const BASE: u64 = 0x180000000;
const SLOT: u64 = crate::vm::DEFAULT_STACK_BASE - 0x100;

fn fixture() -> PeFile {
    PeFile { data: vec![0x90; 0x1200], image_base: BASE,
        entry_point_rva: 0x1000, size_of_image: 0x2000,
        sections: vec![Section { name: ".text".into(), virtual_address: 0x1000,
            virtual_size: 0x1000, raw_address: 0x200, raw_size: 0x1000,
            characteristics: 0x60000020 }], opt_header_offset: 0,
        section_table_offset: 0, file_alignment: 0x200, section_alignment: 0x1000,
        size_of_headers: 0x200, loader_bound: vec![] }
}

#[test]
fn substitution_generation_old_read_must_not_see_later_store() {
    let pe = fixture();
    let mut e = Emulator::new(&pe, crate::vm::DEFAULT_STACK_BASE);
    let address = e.arena.init_reg(Reg::Rcx);
    let old = e.load(address, Width::W64, BASE + 0x1000);
    let slot = e.arena.constant(SLOT, Width::W64);
    let new = e.arena.constant(99, Width::W64);
    e.store(slot, new, Width::W64, BASE + 0x1008);
    e.pins.push((address, SLOT));
    let rewritten = e.substitute(old);
    assert!(matches!(e.arena.op(rewritten), Op::Load(..)),
        "a read taken before the store must retain its old memory version, got {:?}", e.arena.op(rewritten));
    assert_ne!(rewritten, new);
}

#[test]
fn substitution_generation_nonalias_store_allows_prior_fact() {
    let pe = fixture();
    let mut e = Emulator::new(&pe, crate::vm::DEFAULT_STACK_BASE);
    let slot = e.arena.constant(SLOT, Width::W64);
    let ten = e.arena.constant(10, Width::W64);
    e.store(slot, ten, Width::W64, BASE + 0x1000);
    let address = e.arena.init_reg(Reg::Rcx);
    let old = e.load(address, Width::W64, BASE + 0x1008);
    let other = e.arena.constant(SLOT + 16, Width::W64);
    let new = e.arena.constant(99, Width::W64);
    e.store(other, new, Width::W64, BASE + 0x1010);
    e.pins.push((address, SLOT));
    let rewritten = e.substitute(old);
    assert_eq!(e.arena.as_const(rewritten), Some(10));
}

#[test]
fn substitution_generation_partial_store_must_not_rewrite_old_read() {
    let pe = fixture();
    let mut e = Emulator::new(&pe, crate::vm::DEFAULT_STACK_BASE);
    let slot = e.arena.constant(SLOT, Width::W64);
    let ten = e.arena.constant(10, Width::W64);
    e.store(slot, ten, Width::W64, BASE + 0x1000);
    let address = e.arena.init_reg(Reg::Rcx);
    let old = e.load(address, Width::W64, BASE + 0x1008);
    let part = e.arena.constant(SLOT + 7, Width::W64);
    let new = e.arena.constant(99, Width::W8);
    e.store(part, new, Width::W8, BASE + 0x1010);
    e.pins.push((address, SLOT));
    let rewritten = e.substitute(old);
    assert!(matches!(e.arena.op(rewritten), Op::Load(..)),
        "partial overwrite may not change the historical qword read, got {:?}", e.arena.op(rewritten));
}

#[test]
fn substitution_generation_real_instructions_keep_pre_store_read() {
    use iced_x86::code_asm::*;
    let mut pe = fixture();
    let mut a = CodeAssembler::new(64).unwrap();
    a.mov(rax, qword_ptr(rcx)).unwrap();
    a.mov(qword_ptr(rsp - 0x100), 99).unwrap();
    a.ret().unwrap();
    let code = a.assemble(BASE + 0x1000).unwrap();
    pe.data[0x200..0x200 + code.len()].copy_from_slice(&code);
    let mut e = Emulator::new(&pe, crate::vm::DEFAULT_STACK_BASE);
    let pointer = e.state.reg(Reg::Rcx);
    assert!(matches!(e.run(BASE + 0x1000, 2), Stop::Budget { .. }));
    let old = e.state.reg(Reg::Rax);
    e.pins.push((pointer, SLOT));
    let rewritten = e.substitute(old);
    assert!(matches!(e.arena.op(rewritten), Op::Load(..)),
        "RAX read before the store must not change to 99, got {:?}", e.arena.op(rewritten));
}

#[test]
fn substitution_generation_immutable_table_still_folds() {
    let pe = fixture();
    let mut e = Emulator::new(&pe, crate::vm::DEFAULT_STACK_BASE);
    let address = e.arena.init_reg(Reg::Rcx);
    let old = e.load(address, Width::W32, BASE + 0x1000);
    let slot = e.arena.constant(SLOT, Width::W64);
    let new = e.arena.constant(99, Width::W64);
    e.store(slot, new, Width::W64, BASE + 0x1008);
    e.pins.push((address, BASE + 0x1800));
    let rewritten = e.substitute(old);
    assert_eq!(e.arena.as_const(rewritten), Some(0x90909090));
}

#[test]
fn substitution_generation_review_unknown_store_blocks_historical_fold() {
    let pe = fixture();
    let mut e = Emulator::new(&pe, crate::vm::DEFAULT_STACK_BASE);
    let address = e.arena.init_reg(Reg::Rcx);
    let old = e.load(address, Width::W32, BASE + 0x1000);
    let unknown = e.arena.init_reg(Reg::Rdx);
    let later = e.arena.constant(0x12345678, Width::W32);
    e.store(unknown, later, Width::W32, BASE + 0x1008);
    e.pins.push((address, BASE + 0x1800));
    let rewritten = e.substitute(old);
    assert!(matches!(e.arena.op(rewritten), Op::Load(..)),
        "unknown store might alias the pinned readonly-image address");
}

#[test]
fn substitution_generation_review_same_cell_other_offset_is_conservative() {
    let pe = fixture();
    let mut e = Emulator::new(&pe, crate::vm::DEFAULT_STACK_BASE);
    let slot = e.arena.constant(SLOT, Width::W64);
    let old_byte = e.arena.constant(10, Width::W8);
    e.store(slot, old_byte, Width::W8, BASE + 0x1000);
    let address = e.arena.init_reg(Reg::Rcx);
    let old = e.load(address, Width::W8, BASE + 0x1008);
    let neighbor = e.arena.constant(SLOT + 7, Width::W64);
    let later = e.arena.constant(99, Width::W8);
    e.store(neighbor, later, Width::W8, BASE + 0x1010);
    e.pins.push((address, SLOT));
    let rewritten = e.substitute(old);
    assert!(matches!(e.arena.op(rewritten), Op::Load(..)),
        "the deliberately coarse 8-byte generation cell must remain conservative");
}

#[test]
fn substitution_generation_review_cloned_branches_keep_write_history_after_compaction() {
    let pe = fixture();
    let mut initial = Emulator::new(&pe, crate::vm::DEFAULT_STACK_BASE);
    let slot = initial.arena.constant(SLOT, Width::W64);
    let ten = initial.arena.constant(10, Width::W64);
    initial.store(slot, ten, Width::W64, BASE + 0x1000);
    let pointer = initial.arena.init_reg(Reg::Rcx);
    let old = initial.load(pointer, Width::W64, BASE + 0x1008);
    for aliases in [false, true] {
        let mut branch = initial.clone();
        let changed_slot = branch.arena.constant(SLOT + if aliases { 0 } else { 16 }, Width::W64);
        let later = branch.arena.constant(99, Width::W64);
        branch.store(changed_slot, later, Width::W64, BASE + 0x1010);
        let mut retained = [old, pointer];
        branch.compact_arena(&mut retained);
        branch.pins.push((retained[1], SLOT));
        let rewritten = branch.substitute(retained[0]);
        if aliases {
            assert!(matches!(branch.arena.op(rewritten), Op::Load(..)));
        } else {
            assert_eq!(branch.arena.as_const(rewritten), Some(10));
        }
    }
}

#[test]
fn substitution_generation_review_guest_seed_must_not_rewrite_old_read() {
    let pe = fixture();
    let mut e = Emulator::new(&pe, crate::vm::DEFAULT_STACK_BASE);
    let pointer = e.arena.init_reg(Reg::Rcx);
    let old = e.load(pointer, Width::W64, BASE + 0x1000);
    e.guest_ctx = Some(SLOT);
    e.guest_layout = Some(LAYOUTS[0]);
    let params = e.seed_guest_params(crate::ir::expr::BlockRef(91));
    assert_eq!(params.len(), 16, "the fixture must actually seed the guest image");
    let address = LAYOUTS[0].slot(SLOT, Reg::Rax).unwrap();
    e.pins.push((pointer, address));
    let rewritten = e.substitute(old);
    assert!(matches!(e.arena.op(rewritten), Op::Load(..)),
        "an old read must not acquire the new block parameter, got {:?}", e.arena.op(rewritten));
}
