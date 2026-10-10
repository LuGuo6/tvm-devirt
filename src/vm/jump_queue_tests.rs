//! End-to-end work queue regressions using decoded x86 table dispatches.
use super::*;
use crate::binary::pe::Section;
use iced_x86::code_asm::*;

const ENTRY: u64 = 0x1000;
const ARM: u64 = 0x1100;
const SECOND: u64 = 0x1200;
const EXIT: u64 = 0x1300;
const OTHER: u64 = 0x1400;

fn id(handler: u64) -> BlockId { BlockId { handler, vip: None } }

fn image() -> PeFile {
    PeFile {
        data: vec![0u8; 0x900], image_base: 0, entry_point_rva: ENTRY as u32,
        size_of_image: 0x3000,
        sections: vec![
            Section { name: ".text".into(), virtual_address: 0x1000,
                virtual_size: 0x500, raw_address: 0x200, raw_size: 0x500,
                characteristics: 0x6000_0020 },
            Section { name: ".rdata".into(), virtual_address: 0x2000,
                virtual_size: 0x200, raw_address: 0x700, raw_size: 0x200,
                characteristics: 0x4000_0040 },
        ],
        opt_header_offset: 0, section_table_offset: 0, file_alignment: 0x100,
        section_alignment: 0x1000, size_of_headers: 0x200, loader_bound: vec![],
    }
}

fn put(pe: &mut PeFile, address: u64, assemble: impl FnOnce(&mut CodeAssembler)) {
    let mut code = CodeAssembler::new(64).unwrap();
    assemble(&mut code);
    let bytes = code.assemble(address).unwrap();
    let offset = (address - 0x1000 + 0x200) as usize;
    pe.data[offset..offset + bytes.len()].copy_from_slice(&bytes);
}

fn table(pe: &mut PeFile, address: u64, entries: &[u64]) {
    let offset = (address - 0x2000 + 0x700) as usize;
    for (i, value) in entries.iter().enumerate() {
        pe.data[offset + 8 * i..offset + 8 * (i + 1)].copy_from_slice(&value.to_le_bytes());
    }
}

fn dispatch(pe: &mut PeFile, address: u64, table: u64) {
    put(pe, address, |a| {
        a.mov(rax, rcx).unwrap();
        a.and(rax, 3).unwrap();
        a.mov(r11, table).unwrap();
        a.mov(rax, qword_ptr(r11 + rax * 8)).unwrap();
        a.jmp(rax).unwrap();
    });
}

fn explorer(pe: &PeFile) -> Explorer<'_> {
    let mut ex = Explorer::new(pe, crate::vm::DEFAULT_STACK_BASE);
    ex.step_budget = 1000;
    ex.block_budget = 30;
    ex
}

#[test]
fn jump_queue_single_destination_dispatch_has_no_phantom_backedge() {
    let mut pe = image();
    dispatch(&mut pe, ENTRY, 0x2000);
    table(&mut pe, 0x2000, &[ARM; 4]);
    put(&mut pe, ARM, |a| { a.ret().unwrap(); });
    let cfg = explorer(&pe).recover(ENTRY);
    assert_eq!(cfg.unresolved, 0);
    assert_eq!(cfg.blocks.len(), 2);
    assert_eq!(cfg.block(id(ARM)).unwrap().preds, vec![id(ENTRY)]);
    assert!(cfg.back_edges.is_empty(), "one forward Jump must enqueue one continuation: {:?}", cfg.back_edges);
    assert!(constraint_barriers(&cfg).is_empty());
}

#[test]
fn jump_queue_equal_two_way_destinations_enqueue_once() {
    let mut pe = image();
    table(&mut pe, 0x2000, &[ARM; 2]);
    let mut emu = Emulator::new(&pe, crate::vm::DEFAULT_STACK_BASE);
    let input = emu.arena.init_reg(Reg::Rcx);
    let one = emu.arena.constant(1, Width::W64);
    let selector = emu.arena.bin(expr::BinOp::And, input, one);
    let eight = emu.arena.constant(8, Width::W64);
    let offset = emu.arena.bin(expr::BinOp::Mul, selector, eight);
    let base = emu.arena.constant(0x2000, Width::W64);
    let address = emu.arena.bin(expr::BinOp::Add, base, offset);
    let dest = emu.arena.load(address, Width::W64);
    let mut task = Task { emu, value_sets: vec![], resume: ENTRY, id: id(ENTRY),
        from: None, incoming_homes: None, ancestors: vec![] };
    let mut queue = vec![];
    let terminator = explorer(&pe).split_two_way(&mut task, ENTRY, dest, &[selector], &mut queue);
    assert!(matches!(terminator, Some(Terminator::Jump(t)) if t == id(ARM)));
    assert_eq!(queue.len(), 1, "split_two_way owns exactly one continuation");
    assert!(queue[0].emu.pins.is_empty(), "equal destinations prove no predicate value");
    assert_eq!(queue[0].from, Some(id(ENTRY)));
    assert_eq!(queue[0].id, id(ARM));
}

#[test]
fn jump_queue_adopted_jump_has_one_forward_continuation() {
    let mut pe = image();
    put(&mut pe, ENTRY, |a| { a.mov(rax, rcx).unwrap(); a.jmp(rax).unwrap(); });
    put(&mut pe, ARM, |a| { a.ret().unwrap(); });
    let ex = explorer(&pe);
    let barriers = HashSet::new();
    let mut known = ex.recover_pass_with(ENTRY, &HashSet::new(), None, None, Some(&barriers));
    let entry = known.blocks.iter_mut().find(|b| b.id == id(ENTRY)).unwrap();
    assert!(matches!(entry.terminator, Terminator::Unresolved { .. }));
    // Supply only the known edge for the actual adoption wiring test. This
    // fixture does not assert that an arbitrary RCX is semantically ARM.
    entry.terminator = Terminator::Jump(id(ARM));
    let cfg = ex.recover_pass_with(ENTRY, &HashSet::new(), Some(&known), None, Some(&barriers));
    assert_eq!(cfg.unresolved, 0);
    assert_eq!(cfg.blocks.len(), 2);
    assert!(matches!(cfg.block(id(ENTRY)).unwrap().terminator, Terminator::Jump(t) if t == id(ARM)));
    assert!(cfg.back_edges.is_empty());
    assert!(constraint_barriers(&cfg).is_empty());
}

#[test]
fn jump_queue_single_destination_preserves_proven_selector_set_downstream() {
    let mut pe = image();
    dispatch(&mut pe, ENTRY, 0x2000);
    dispatch(&mut pe, ARM, 0x2040);
    dispatch(&mut pe, SECOND, 0x2080);
    table(&mut pe, 0x2000, &[ARM, OTHER, ARM, OTHER]);
    table(&mut pe, 0x2040, &[SECOND, 0x2000, SECOND, 0x2000]);
    table(&mut pe, 0x2080, &[EXIT, 0x2000, EXIT, 0x2000]);
    put(&mut pe, EXIT, |a| { a.ret().unwrap(); });
    put(&mut pe, OTHER, |a| { a.ret().unwrap(); });
    let cfg = explorer(&pe).recover(ENTRY);
    assert_eq!(cfg.unresolved, 0, "odd selector values are proved absent on ARM and SECOND");
    assert_eq!(cfg.blocks.len(), 5);
    assert!(cfg.back_edges.is_empty(), "forward single-target table dispatches are not loops");
}

#[test]
fn jump_queue_real_loop_keeps_a_constraint_barrier() {
    let mut pe = image();
    put(&mut pe, ENTRY, |a| {
        a.test(rcx, rcx).unwrap();
        a.je(ARM).unwrap();
        a.ret().unwrap();
    });
    put(&mut pe, ARM, |a| {
        a.test(rdx, rdx).unwrap();
        a.je(ARM).unwrap();
        a.ret().unwrap();
    });
    let cfg = explorer(&pe).recover(ENTRY);
    assert!(cfg.back_edges.iter().any(|(_, to)| *to == id(ARM)), "genuine loop remains explicit");
    assert!(constraint_barriers(&cfg).contains(&id(ARM)));
}

// Supply known edges only to exercise adoption's queue contract. This does not
// claim that arbitrary RCX inputs semantically have these branch targets.
fn adopted_successors(
    pe: &PeFile,
    terminator: impl FnOnce(&mut expr::Arena) -> Terminator,
) -> (Terminator, Vec<BlockId>) {
    let ex = explorer(pe);
    let mut known = ex.recover_pass_with(ENTRY, &HashSet::new(), None, None, None);
    let term = terminator(&mut known.arena);
    let previous = known.blocks.iter_mut().find(|b| b.id == id(ENTRY)).unwrap();
    previous.terminator = term;
    let cost = previous.cost;
    let mut emu = Emulator::new(pe, crate::vm::DEFAULT_STACK_BASE);
    let selector = emu.arena.init_reg(Reg::Rcx);
    let mut task = Task { emu, value_sets: vec![(selector, 0b1010)], resume: ENTRY,
        id: id(ENTRY), from: None, incoming_homes: None, ancestors: vec![] };
    let mut queue = vec![];
    let adopted = ex.adopt_known(&mut task, Some(&known), &mut queue, cost).unwrap();
    for successor in &queue {
        assert_eq!(successor.from, Some(id(ENTRY)));
        assert_eq!(successor.value_sets, task.value_sets);
        assert_eq!(successor.emu.pins, task.emu.pins,
            "adoption adds no predicate fact from the previous arena");
        assert_eq!(successor.ancestors, vec![id(ENTRY)]);
    }
    (adopted, queue.iter().map(|t| t.id).collect())
}

#[test]
fn jump_queue_adopted_switch_queues_distinct_full_ids_and_preserves_case_slots() {
    let mut pe = image();
    put(&mut pe, ENTRY, |a| { a.mov(rax, rcx).unwrap(); a.jmp(rax).unwrap(); });
    let arm_vip1 = BlockId { handler: ARM, vip: Some(1) };
    let arm_vip2 = BlockId { handler: ARM, vip: Some(2) };
    let targets = vec![id(ARM), arm_vip1, id(ARM), arm_vip2, arm_vip1, id(SECOND)];
    let (term, queued) = adopted_successors(&pe, |arena| Terminator::Switch {
        selector: arena.init_reg(Reg::Rcx), targets: targets.clone(),
    });
    assert_eq!(queued, vec![id(ARM), arm_vip1, arm_vip2, id(SECOND)],
        "identical snapshots enqueue once per full BlockId, in first-seen order");
    assert!(matches!(term, Terminator::AdoptedSwitch { targets: slots } if slots == targets),
        "the IR must retain every original selector case slot");
}

#[test]
fn jump_queue_adopted_equal_branch_queues_one_snapshot() {
    let mut pe = image();
    put(&mut pe, ENTRY, |a| { a.mov(rax, rcx).unwrap(); a.jmp(rax).unwrap(); });
    let (term, queued) = adopted_successors(&pe, |arena| Terminator::Branch {
        predicate: arena.init_reg(Reg::Rcx), taken: id(ARM), not_taken: id(ARM),
    });
    assert_eq!(queued, vec![id(ARM)]);
    assert!(matches!(term, Terminator::Adopted { taken, not_taken }
        if taken == id(ARM) && not_taken == id(ARM)));
}

#[test]
fn jump_queue_adopted_branch_same_handler_different_vip_keeps_both_snapshots() {
    let mut pe = image();
    put(&mut pe, ENTRY, |a| { a.mov(rax, rcx).unwrap(); a.jmp(rax).unwrap(); });
    let taken = BlockId { handler: ARM, vip: Some(1) };
    let not_taken = BlockId { handler: ARM, vip: Some(2) };
    let (term, queued) = adopted_successors(&pe, |arena| Terminator::Branch {
        predicate: arena.init_reg(Reg::Rcx), taken, not_taken,
    });
    assert_eq!(queued, vec![taken, not_taken]);
    assert!(matches!(term, Terminator::Adopted { taken: t, not_taken: n }
        if t == taken && n == not_taken));
}
