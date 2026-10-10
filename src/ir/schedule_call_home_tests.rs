use super::*;
use crate::ir::expr::{BlockRef, CallOrigin};

fn sample() -> Cfg {
    let mut arena = Arena::new();
    let first = BlockId { handler: 0x1000, vip: None };
    let next = BlockId { handler: 0x2000, vip: None };
    let origin = CallOrigin { block: first, event_index: 0, site: 0x1010 };
    let old = arena.opaque("call_ret", Width::W64);
    arena.bind_call_origin(old, origin);
    // A successor is grafted through its own memo and gets a fresh opaque Ref.
    let copy = arena.opaque("call_ret", Width::W64);
    arena.bind_call_origin(copy, origin);
    let new = arena.opaque("call_ret", Width::W64);
    let a = arena.constant(0x5000, Width::W64);
    let b = arena.constant(0x6000, Width::W64);
    let c = arena.constant(0x7000, Width::W64);
    let rsp = arena.constant(0x700000000, Width::W64);
    let dest = arena.constant(0, Width::W64);
    let call = |target, site, ret, args| Event::Call {
        target: Some(target), site, rsp: None, import_slot: None, ret, args,
    };
    Cfg {
        entry: first, arena, total_steps: 0, unresolved: 0, back_edges: vec![],
        trivial_phis_removed: 0, timed_out: false, stack_base: 0x700000000,
        native_entry_proofs: Default::default(), vm_range: (0, 0), blocks: vec![
            Block { id: first, block_ref: BlockRef(0), params: vec![],
                from_vm_context: false, cost: 0, preds: vec![],
                events: vec![call(a, 0x1010, Some(old), vec![]),
                    call(b, 0x1020, Some(new), vec![])],
                exit_regs: vec![(Reg::Rsp, rsp)], terminator: Terminator::Jump(next) },
            Block { id: next, block_ref: BlockRef(1), params: vec![],
                from_vm_context: false, cost: 0, preds: vec![first],
                events: vec![call(c, 0x2010, None, vec![(Reg::Rcx, copy)])],
                exit_regs: vec![(Reg::Rsp, rsp)], terminator: Terminator::Return { dest } },
        ],
    }
}

#[test]
fn call_home_schedule_commits_old_return_before_another_call_overwrites_rax() {
    let graph = sample();
    let blocks = schedule(&graph);
    let owner = &blocks[0];
    let calls: Vec<_> = owner.instrs.iter().enumerate()
        .filter_map(|(i, op)| matches!(op, Instr::Call { .. }).then_some(i)).collect();
    assert_eq!(calls.len(), 2);
    let (write, value) = owner.instrs.iter().enumerate().find_map(|(i, op)| {
        if let Instr::WriteCallHome { slot: 0, value: Operand::Val(value) } = op {
            Some((i, *value))
        } else { None }
    }).expect("cross-block store must survive local DCE");
    assert!(calls[0] < write && write < calls[1]);
    assert!(owner.instrs[calls[0] + 1..write].iter().any(|op| matches!(op,
        Instr::Opaque { dst, tag: "call_ret", at: Some(Reg::Rax), .. } if *dst == value)));
    let next = &blocks[1];
    let read = next.instrs.iter().find_map(|op| match op {
        Instr::ReadCallHome { dst, slot: 0 } => Some(*dst), _ => None,
    }).expect("raw grafted return needs the proven private slot");
    assert!(next.instrs.iter().any(|op| matches!(op, Instr::Call { args, .. }
        if args.contains(&(Reg::Rcx, Operand::Val(read))))));
    assert!(!next.instrs.iter().any(|op| matches!(op,
        Instr::Opaque { tag: "call_ret", at: None, .. })));
}

#[test]
fn call_home_schedule_does_not_allocate_for_an_unproven_join() {
    let mut graph = sample();
    let mut bypass = graph.blocks[0].clone();
    bypass.id.handler = 0x3000;
    bypass.block_ref = BlockRef(2);
    bypass.events.clear();
    graph.blocks.push(bypass);
    let blocks = schedule(&graph);
    assert!(!blocks.iter().flat_map(|b| &b.instrs).any(|op| matches!(op,
        Instr::WriteCallHome { .. } | Instr::ReadCallHome { .. })));
    assert!(blocks[1].instrs.iter().any(|op| matches!(op,
        Instr::Opaque { tag: "call_ret", at: None, .. })));
}
