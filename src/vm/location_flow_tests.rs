//! Wiring regressions through the actual x86 evaluator and recovery work queue.
//! These deliberately reject their whole-state proof: the round-trip fixture
//! retains historical scratch and flags. Positive seeding is tested separately
//! in ir/location_home_tests.rs, without claiming a complete accepted CFG here.
use super::*;
use crate::ir::lift::context_tests::{prepared_locations, round_trip};
use iced_x86::code_asm::*;

const IMAGE: u64 = 0x180000000;
const ENTRY: u64 = IMAGE + 0x2000;
const ARM: u64 = IMAGE + 0x2200;
const JOIN: u64 = IMAGE + 0x2400;
const STACK: u64 = 0x7fff00003000;

fn id(handler: u64) -> BlockId {
    BlockId { handler, vip: None }
}

fn put(pe: &mut PeFile, address: u64, assemble: impl FnOnce(&mut CodeAssembler)) {
    let mut code = CodeAssembler::new(64).unwrap();
    assemble(&mut code);
    let bytes = code.assemble(address).unwrap();
    let offset = (address - IMAGE - 0x1000 + 0x200) as usize;
    pe.data[offset..offset + bytes.len()].copy_from_slice(&bytes);
}

fn explorer(pe: &PeFile) -> Explorer<'_> {
    let mut seed = prepared_locations(pe);
    seed.run(IMAGE + 0x1000, 1000);
    assert_eq!(seed.guest_registers().unwrap().len(), 16);
    let mut ex = Explorer::new(pe, STACK);
    ex.step_budget = 1000;
    ex.block_budget = 30;
    ex.test_seed = Some(seed);
    ex
}

fn assert_parent_snapshot(ex: &Explorer<'_>, cfg: &Cfg, observation: &LocationObservation) {
    assert_eq!(
        observation.homes.len(),
        16,
        "actual parent snapshot is required"
    );
    let parent = cfg.block(observation.from.unwrap()).unwrap();
    assert!(parent.from_vm_context);
    let captured = ex.location_exit_snapshots.borrow();
    let (raw, lowered) = captured.get(&parent.id).unwrap();
    assert_eq!(
        raw, &observation.homes,
        "queued homes must be the captured parent exit"
    );
    for &(reg, ref value) in &observation.homes {
        let output = parent.exit_regs.iter().find(|(r, _)| *r == reg).unwrap().1;
        let mapped = &lowered.iter().find(|(r, _)| *r == reg).unwrap().1;
        assert_eq!(
            &expr::render(&cfg.arena, output, 12),
            mapped,
            "parent {} and incoming {} disagree for {}",
            parent.id,
            observation.block,
            reg.name()
        );
        // Pure expressions do not require the shared graft memo. Historical
        // dynamic values may legitimately lower to this parent's own Param.
        if reg == Reg::Rsp || reg == Reg::R12 {
            assert_eq!(mapped, value);
        }
    }
}

#[test]
fn location_flow_late_diamond_observes_both_parent_snapshots_before_visited_pruning() {
    let mut pe = round_trip();
    put(&mut pe, ENTRY, |a| {
        a.test(r12, r12).unwrap();
        a.je(ARM).unwrap();
        a.test(r13, r13).unwrap();
        a.je(JOIN).unwrap();
        a.ret().unwrap();
    });
    put(&mut pe, ARM, |a| {
        a.test(r14, r14).unwrap();
        a.je(JOIN).unwrap();
        a.ret().unwrap();
    });
    put(&mut pe, JOIN, |a| {
        a.ret().unwrap();
    });
    let ex = explorer(&pe);
    let cfg = ex.recover_pass_with(ENTRY, &HashSet::from([id(JOIN)]), None, None, None);
    let observations = ex.location_observations.borrow();
    assert_eq!(
        observations.len(),
        2,
        "late edge must reach the real observer"
    );
    assert!(!observations[0].late);
    assert!(observations[1].late);
    assert_ne!(observations[0].from, observations[1].from);
    assert_eq!(cfg.block(id(JOIN)).unwrap().preds.len(), 2);
    for observation in observations.iter() {
        assert_eq!(observation.block, id(JOIN));
        assert!(!observation.backedge);
        assert!(observation.rejected);
        assert_parent_snapshot(&ex, &cfg, observation);
    }
    assert!(
        cfg.unresolved > 0,
        "a failed late proof must survive final CFG construction"
    );
}

fn loop_pe(symbolic_backedge: bool) -> PeFile {
    let mut pe = round_trip();
    put(&mut pe, ENTRY, |a| {
        a.test(r12, r12).unwrap();
        a.je(ARM).unwrap();
        a.ret().unwrap();
    });
    put(&mut pe, ARM, |a| {
        a.test(r13, r13).unwrap();
        a.je(JOIN).unwrap();
        if symbolic_backedge {
            // No candidate executable destination can be derived. The known
            // concrete-pass shape below supplies an adopted Backedge instead.
            a.xor(rax, rbx).unwrap();
            a.jmp(rax).unwrap();
        } else {
            // A genuine guest store makes the interpreter's revisit guard
            // observe progress before the resolved indirect jump to ARM.
            a.mov(rdx, STACK + 0x10000).unwrap();
            a.mov(qword_ptr(rdx), r12).unwrap();
            a.mov(r11, ARM).unwrap();
            a.jmp(r11).unwrap();
        }
    });
    put(&mut pe, JOIN, |a| {
        a.ret().unwrap();
    });
    pe
}

#[test]
fn location_flow_explicit_backedge_rechecks_the_actual_loop_exit_snapshot() {
    let pe = loop_pe(false);
    let ex = explorer(&pe);
    let cfg = ex.recover_pass_with(ENTRY, &HashSet::from([id(ARM)]), None, None, None);
    let observations = ex.location_observations.borrow();
    assert_eq!(observations.len(), 2);
    let loop_edge = observations
        .iter()
        .find(|o| o.backedge)
        .expect("real Stop::Backedge path");
    assert!(loop_edge.late && loop_edge.rejected);
    assert!(
        matches!(cfg.block(loop_edge.from.unwrap()).unwrap().terminator,
        Terminator::Backedge { target } if target == id(ARM))
    );
    assert!(cfg.back_edges.contains(&(loop_edge.from.unwrap(), id(ARM))));
    assert_parent_snapshot(&ex, &cfg, loop_edge);
    assert!(cfg.unresolved > 0);
}

#[test]
fn location_flow_adopted_backedge_uses_the_same_actual_incoming_proof() {
    let pe = loop_pe(true);
    let ex = explorer(&pe);
    let mut known = ex.recover_pass_with(ENTRY, &HashSet::new(), None, None, None);
    let body = known
        .blocks
        .iter_mut()
        .find(|block| matches!(block.terminator, Terminator::Unresolved { .. }))
        .expect("symbolic destination must need adoption");
    let body_id = body.id;
    body.terminator = Terminator::Backedge { target: id(ARM) };
    ex.location_observations.borrow_mut().clear();
    let cfg = ex.recover_pass_with(ENTRY, &HashSet::from([id(ARM)]), Some(&known), None, None);
    let observations = ex.location_observations.borrow();
    let adopted = observations
        .iter()
        .find(|o| o.backedge)
        .expect("adopt_known Backedge path");
    assert_eq!(adopted.from, Some(body_id));
    assert_eq!(adopted.block, id(ARM));
    assert!(adopted.late && adopted.rejected);
    assert!(matches!(cfg.block(body_id).unwrap().terminator,
        Terminator::Backedge { target } if target == id(ARM)));
    assert_parent_snapshot(&ex, &cfg, adopted);
    assert!(cfg.unresolved > 0);
}

#[test]
fn location_flow_fixed_point_replays_still_reject_discovered_join() {
    let pe = loop_pe(false);
    let ex = explorer(&pe);
    let cfg = ex.recover(ENTRY);
    let observations = ex.location_observations.borrow();
    assert!(
        observations
            .iter()
            .any(|o| o.block == id(ARM) && o.backedge)
    );
    assert!(observations.iter().all(|o| o.rejected));
    assert!(cfg.unresolved > 0);
}
