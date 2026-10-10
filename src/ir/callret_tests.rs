use super::*;
use crate::ir::expr::{Arena, BlockRef};

fn bid(handler: u64) -> BlockId {
    BlockId { handler, vip: None }
}

fn saved_return_cfg() -> Cfg {
    let mut arena = Arena::new();
    let object = arena.opaque("call_ret", Width::W64);
    let later = arena.opaque("call_ret", Width::W64);
    let parameter = arena.param(BlockRef(1), Reg::Rdi);
    let target_a = arena.constant(0x5000, Width::W64);
    let target_b = arena.constant(0x6000, Width::W64);
    let target_c = arena.constant(0x7000, Width::W64);
    let dest = arena.constant(0, Width::W64);
    let rsp = arena.constant(0x7ff0_0000, Width::W64);
    let call = |target, ret, args| Event::Call {
        target: Some(target),
        site: 0x1000,
        rsp: Some(0x7fef_ffd8),
        import_slot: None,
        args,
        ret,
    };
    Cfg {
        entry: bid(0x1000),
        arena,
        total_steps: 0,
        unresolved: 0,
        back_edges: vec![],
        trivial_phis_removed: 0,
        timed_out: false,
        stack_base: 0x7ff0_0000,
        native_entry_proofs: Default::default(), vm_range: (0, 0),
        blocks: vec![
            Block {
                id: bid(0x1000),
                block_ref: BlockRef(0),
                params: vec![],
                events: vec![
                    call(target_a, Some(object), vec![]),
                    call(target_b, Some(later), vec![]),
                ],
                from_vm_context: true,
                exit_regs: vec![(Reg::Rax, later), (Reg::Rdi, object), (Reg::Rsp, rsp)],
                terminator: Terminator::Jump(bid(0x2000)),
                cost: 0,
                preds: vec![],
            },
            Block {
                id: bid(0x2000),
                block_ref: BlockRef(1),
                params: vec![(Reg::Rdi, parameter)],
                events: vec![call(target_c, None, vec![(Reg::Rcx, parameter)])],
                from_vm_context: true,
                exit_regs: vec![(Reg::Rsp, rsp)],
                terminator: Terminator::Return { dest },
                cost: 0,
                preds: vec![bid(0x1000)],
            },
        ],
    }
}

#[test]
fn saved_call_return_keeps_its_cross_block_home_after_phi_simplification() {
    let mut cfg = saved_return_cfg();
    cfg.simplify_phis();
    cfg.remove_dead_params();
    let blocks = schedule(&cfg);
    let successor = blocks.iter().find(|b| b.id == bid(0x2000)).unwrap();
    let arg = successor
        .instrs
        .iter()
        .find_map(|i| match i {
            Instr::Call { args, .. } => args.iter().find(|(r, _)| *r == Reg::Rcx).map(|(_, o)| *o),
            _ => None,
        })
        .unwrap();
    assert!(
        matches!(arg, Operand::Param(Reg::Rdi) | Operand::Entry(Reg::Rdi)),
        "object saved in RDI must not become the later RAX return: {arg:?}"
    );
    assert!(
        blocks[0].exits.iter().any(|(r, _)| *r == Reg::Rdi),
        "the predecessor must materialize the live object's RDI home"
    );
}

#[test]
fn physical_parameter_is_not_renamed_as_a_function_entry_value() {
    let mut cfg = saved_return_cfg();
    let original_rdi = cfg.arena.init_reg(Reg::Rdi);
    cfg.blocks[0].exit_regs.push((Reg::R12, original_rdi));
    let parameter = cfg.blocks[1].params[0].1;
    cfg.blocks[1].exit_regs.push((Reg::Rdi, parameter));
    let blocks = schedule(&cfg);
    let successor = blocks.iter().find(|b| b.id == bid(0x2000)).unwrap();
    let arg = successor
        .instrs
        .iter()
        .find_map(|i| match i {
            Instr::Call { args, .. } => args.iter().find(|(r, _)| *r == Reg::Rcx).map(|(_, o)| *o),
            _ => None,
        })
        .unwrap();
    assert!(
        matches!(arg, Operand::Param(Reg::Rdi) | Operand::Entry(Reg::Rdi)),
        "block-entry RDI is not the function-entry RDI now held in R12: {arg:?}"
    );
}

#[test]
fn multiple_predecessors_supply_the_saved_home_not_their_latest_return() {
    let mut cfg = saved_return_cfg();
    let first_object = cfg.blocks[0].exit_regs[1].1;
    let other_object = cfg.arena.opaque("call_ret", Width::W64);
    let other_later = cfg.arena.opaque("call_ret", Width::W64);
    let target = cfg.arena.constant(0x8000, Width::W64);
    let mut other = cfg.blocks[0].clone();
    other.id = bid(0x3000);
    other.block_ref = BlockRef(2);
    for (index, event) in other.events.iter_mut().enumerate() {
        if let Event::Call {
            ret,
            target: call_target,
            ..
        } = event
        {
            *ret = Some(if index == 0 {
                other_object
            } else {
                other_later
            });
            *call_target = Some(target);
        }
    }
    other.exit_regs[0].1 = other_later;
    other.exit_regs[1].1 = other_object;
    let entry = bid(0x9000);
    cfg.blocks[0].preds = vec![entry];
    other.preds = vec![entry];
    cfg.blocks[1].preds.push(other.id);
    cfg.blocks.push(other);
    let predicate = cfg.arena.init_reg(Reg::Rcx);
    cfg.blocks.insert(
        0,
        Block {
            id: entry,
            block_ref: BlockRef(3),
            params: vec![],
            events: vec![],
            from_vm_context: true,
            exit_regs: vec![],
            terminator: Terminator::Branch {
                predicate,
                taken: bid(0x1000),
                not_taken: bid(0x3000),
            },
            cost: 0,
            preds: vec![],
        },
    );
    cfg.entry = entry;
    cfg.simplify_phis();
    let phi = cfg.phis().into_iter().find(|p| p.reg == Reg::Rdi).unwrap();
    assert_eq!(
        phi.incoming
            .iter()
            .filter_map(|(_, v)| *v)
            .collect::<Vec<_>>(),
        vec![first_object, other_object]
    );
    let scheduled = schedule(&cfg);
    let next = scheduled.iter().find(|b| b.id == bid(0x2000)).unwrap();
    assert!(next.instrs.iter().any(|i| matches!(i,
        Instr::Call { args, .. } if args.contains(&(Reg::Rcx, Operand::Param(Reg::Rdi))))));
    for id in [bid(0x1000), bid(0x3000)] {
        let block = scheduled
            .iter()
            .find(|b| b.id == id)
            .expect("both predecessors are reachable");
        assert!(block.exits.iter().any(|(r, _)| *r == Reg::Rdi));
    }
}

#[test]
fn a_missing_predecessor_does_not_simplify_the_remaining_home() {
    let mut cfg = saved_return_cfg();
    let incoming = cfg.arena.init_reg(Reg::Rdx);
    cfg.blocks[0].exit_regs[1].1 = incoming;
    cfg.blocks[1].preds.push(bid(0x3000));
    cfg.simplify_phis();
    assert_eq!(cfg.blocks[1].params.len(), 1);
}
