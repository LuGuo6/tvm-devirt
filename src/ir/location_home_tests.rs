//! Independent join regressions over executed restore/call/save/copy states.
//! The cleanup below is real code: it removes unrelated physical call clobbers
//! without discarding abstract state or fabricating complete-save provenance.
use super::*;
use crate::vm::state::{Flag, XmmHalf};
use iced_x86::code_asm::*;

const IMAGE: u64 = 0x180000000;
const START: u64 = IMAGE + 0x1000;
const CLEANUP: u64 = IMAGE + 0x1800;
const READ_CANONICAL: u64 = IMAGE + 0x1a00;
const READ_MIRROR: u64 = IMAGE + 0x1a40;
const READ_EXISTING_B: u64 = IMAGE + 0x1a80;
const STACK: u64 = 0x7fff00003000;
const ALT: u64 = STACK - 0x300;
const ACTIVE: u64 = STACK - 0x100;
const EXISTING_B: u64 = STACK - 0x600;

fn put_code(pe: &mut PeFile, at: u64, assembler: &mut CodeAssembler) {
    let code = assembler.assemble(at).unwrap();
    let offset = 0x200 + (at - START) as usize;
    assert!(offset + code.len() < 0x2200);
    pe.data[offset..offset + code.len()].copy_from_slice(&code);
}

fn fixture(canonical_constant: Option<i32>) -> PeFile {
    let mut pe = super::super::context_tests::round_trip();
    let mut cleanup = CodeAssembler::new(64).unwrap();
    if let Some(value) = canonical_constant {
        cleanup
            .mov(
                qword_ptr(rbp + LAYOUTS[1].slot(0, Reg::Rax).unwrap() as i32),
                value,
            )
            .unwrap();
        // This old mirror is deliberately constant and equal across edges.
        // It must not distract from the differing, unplanned existing-B slot.
        cleanup
            .mov(
                qword_ptr(r10 + LAYOUTS[0].slot(0, Reg::Rax).unwrap() as i32),
                0x55,
            )
            .unwrap();
    }
    for machine in [
        rax, rbx, rcx, rdx, rbp, rsi, rdi, r8, r9, r10, r11, r12, r13, r14, r15,
    ] {
        cleanup.mov(machine, 0u64).unwrap();
    }
    cleanup.push(0x202i32).unwrap();
    cleanup.popfq().unwrap();
    cleanup.int3().unwrap();
    put_code(&mut pe, CLEANUP, &mut cleanup);
    for (pc, address) in [
        (READ_CANONICAL, LAYOUTS[1].slot(ALT, Reg::Rax).unwrap()),
        (READ_MIRROR, LAYOUTS[0].slot(ACTIVE, Reg::Rax).unwrap()),
        (
            READ_EXISTING_B,
            LAYOUTS[0].slot(EXISTING_B, Reg::Rax).unwrap(),
        ),
    ] {
        let mut reader = CodeAssembler::new(64).unwrap();
        reader.mov(r8, address).unwrap();
        reader.mov(rax, qword_ptr(r8)).unwrap();
        reader.int3().unwrap();
        put_code(&mut pe, pc, &mut reader);
    }
    pe
}

fn incoming(pe: &PeFile, b_rax: u64, opaque_padding: usize) -> Emulator<'_> {
    let mut emu = super::super::context_tests::prepared(pe);
    emu.diagnostic_context_origin = false;
    emu.diagnostic_location_homes = true;
    // B predates the actual restore/call/save. Its contents may coincide with
    // current homes, but it never acquired a fresh complete-save proof.
    for &reg in LAYOUTS[0].order {
        let value = if reg == Reg::Rax {
            b_rax
        } else if reg == Reg::Rsp {
            STACK
        } else {
            0x7000 + reg as u64
        };
        let value = emu.arena.constant(value, Width::W64);
        emu.state.store_concrete(
            &emu.arena,
            LAYOUTS[0].slot(EXISTING_B, reg).unwrap(),
            value,
            Width::W64,
        );
    }
    for _ in 0..opaque_padding {
        emu.arena.opaque("unreachable_test_history", Width::W64);
    }
    emu.run(START, 1000);
    assert!(emu.context_tracker.saved_images().contains(&(ALT, 1)));
    assert!(emu.context_tracker.saved_images().contains(&(ACTIVE, 0)));
    assert!(
        !emu.context_tracker
            .saved_images()
            .contains(&(EXISTING_B, 0))
    );
    emu.run(CLEANUP, 100);
    assert_eq!(emu.guest_ctx, Some(ALT));
    assert_eq!(emu.guest_layout, Some(LAYOUTS[1]));
    assert!(
        emu.state
            .regs
            .values()
            .all(|value| emu.arena.as_const(*value).is_some())
    );
    assert!(
        emu.state
            .flags
            .values()
            .all(|value| emu.arena.as_const(*value).is_some())
    );
    emu
}

fn homes(emu: &mut Emulator<'_>) -> Vec<(Reg, Ref)> {
    LAYOUTS[1]
        .order
        .iter()
        .map(|&reg| {
            (
                reg,
                emu.raw_slot(LAYOUTS[1].slot(ALT, reg).unwrap()).unwrap(),
            )
        })
        .collect()
}

fn rax_home(homes: &[(Reg, Ref)]) -> Ref {
    homes.iter().find(|(reg, _)| *reg == Reg::Rax).unwrap().1
}

fn read_rax(emu: &mut Emulator<'_>, pc: u64) -> Ref {
    let before = emu.steps;
    emu.run(pc, 10);
    assert!(
        emu.steps >= before + 2,
        "the MOV reader must really execute"
    );
    emu.state.reg(Reg::Rax)
}

fn rejection(result: ProofResult<LocationHomePlan>) -> String {
    match result {
        Ok(_) => panic!("unproven incoming state was accepted"),
        Err(error) => error,
    }
}

#[test]
fn substitution_generation_review_location_seed_must_not_rewrite_old_read() {
    let pe = fixture(None);
    let mut incoming = incoming(&pe, 0x99, 0);
    let saved_homes = homes(&mut incoming);
    let plan = LocationHomePlan::discover(&mut incoming, &saved_homes, &[]).unwrap();
    let pointer = incoming.arena.init_reg(Reg::Rcx);
    let old = incoming.load(pointer, Width::W64, READ_CANONICAL);
    let params = plan.seed(&mut incoming, BlockRef(92));
    assert_eq!(params.len(), 15);
    incoming.pins.push((pointer, LAYOUTS[1].slot(ALT, Reg::Rax).unwrap()));
    let rewritten = incoming.substitute(old);
    assert!(matches!(incoming.arena.op(rewritten), Op::Load(..)),
        "location-home seed must not change an earlier read to a new parameter: {:?}", incoming.arena.op(rewritten));
}

#[test]
fn existing_b_constants_that_match_each_edge_home_are_not_parameterized() {
    let left_pe = fixture(Some(0));
    let right_pe = fixture(Some(1));
    let mut left = incoming(&left_pe, 0, 0);
    let mut right = incoming(&right_pe, 1, 0);
    let left_homes = homes(&mut left);
    let right_homes = homes(&mut right);
    let mut plan = LocationHomePlan::discover(&mut left, &left_homes, &[]).unwrap();
    let b_slot = LAYOUTS[0].slot(EXISTING_B, Reg::Rax).unwrap();
    assert!(
        plan.locations
            .iter()
            .all(|location| location.address != b_slot)
    );
    assert_eq!(left.raw_slot(b_slot), Some(rax_home(&left_homes)));
    assert_eq!(right.raw_slot(b_slot), Some(rax_home(&right_homes)));

    let mut seeded = left.clone();
    plan.seed(&mut seeded, BlockRef(41));
    let old_b = read_rax(&mut seeded, READ_EXISTING_B);
    assert_eq!(seeded.arena.as_const(old_b), Some(0));
    let mut other_edge = right.clone();
    let other_b = read_rax(&mut other_edge, READ_EXISTING_B);
    assert_eq!(other_edge.arena.as_const(other_b), Some(1));
    let error = plan.validate(&mut right, &right_homes, &[]).unwrap_err();
    assert!(
        error.contains(&format!("unseeded incoming memory byte {b_slot:#x}")),
        "{error}"
    );
}

#[test]
fn planned_homes_seed_actual_reads_across_distinct_opaque_histories() {
    let pe = fixture(None);
    let mut first = incoming(&pe, 0x99, 0);
    let mut late = incoming(&pe, 0x99, 11);
    let first_homes = homes(&mut first);
    let late_homes = homes(&mut late);
    assert_ne!(
        first.arena.op(rax_home(&first_homes)),
        late.arena.op(rax_home(&late_homes)),
        "independent call returns must have different arena-local IDs in this fixture"
    );
    let mut plan = LocationHomePlan::discover(&mut first, &first_homes, &[]).unwrap();
    plan.validate(&mut late, &late_homes, &[]).unwrap();
    let rsp_slot = LAYOUTS[1].slot(ALT, Reg::Rsp).unwrap();
    let before_rsp = first.raw_slot(rsp_slot).unwrap();
    let params = plan.seed(&mut first, BlockRef(42));
    assert_eq!(params.len(), 15);
    assert!(!params.iter().any(|(reg, _)| *reg == Reg::Rsp));
    assert_eq!(first.raw_slot(rsp_slot), Some(before_rsp));
    assert_eq!(first.arena.as_const(before_rsp), Some(STACK));
    let parameter = first.arena.param(BlockRef(42), Reg::Rax);
    assert!(first.guest_call_results.contains(&parameter));
    assert!(!first.guest_call_results.contains(&rax_home(&first_homes)));
    let five = first.arena.constant(5, Width::W64);
    let field = first.arena.bin(BinOp::Add, parameter, five);
    assert!(first.is_guest_call_address(field));
    assert_eq!(first.region_of_symbolic(field), Region::Guest);
    for pc in [READ_CANONICAL, READ_MIRROR] {
        let mut consumer = first.clone();
        let value = read_rax(&mut consumer, pc);
        assert_eq!(consumer.arena.op(value), &Op::Param(BlockRef(42), Reg::Rax));
    }
    let mut b_consumer = first.clone();
    let value = read_rax(&mut b_consumer, READ_EXISTING_B);
    assert_eq!(b_consumer.arena.as_const(value), Some(0x99));
}

#[test]
fn identical_opaque_ids_in_independent_arenas_do_not_prove_existing_b() {
    let pe = fixture(None);
    let mut first = incoming(&pe, 0, 0);
    let mut second = incoming(&pe, 0, 0);
    let first_value = first
        .arena
        .opaque("independent_external_result", Width::W64);
    let second_value = second
        .arena
        .opaque("independent_external_result", Width::W64);
    assert_eq!(
        first_value, second_value,
        "the regression requires a Ref-ID collision"
    );
    assert_eq!(first.arena.op(first_value), second.arena.op(second_value));
    let address = LAYOUTS[0].slot(EXISTING_B, Reg::Rax).unwrap();
    for (emu, value) in [(&mut first, first_value), (&mut second, second_value)] {
        emu.state
            .store_concrete(&emu.arena, address, value, Width::W64);
        let snapshot = homes(emu);
        let mut consumer = emu.clone();
        assert_eq!(read_rax(&mut consumer, READ_EXISTING_B), value);
        let error = rejection(LocationHomePlan::discover(emu, &snapshot, &[]));
        assert!(
            error.contains("unseeded memory byte") && error.contains("no cross-edge identity"),
            "{error}"
        );
    }
}

#[test]
fn unmapped_dynamic_machine_state_rejects_the_join() {
    let pe = fixture(None);
    for place in [
        "memory",
        "register",
        "flag",
        "xmm",
        "pin",
        "value_set",
        "symbolic_store",
    ] {
        let mut emu = incoming(&pe, 0, 0);
        let snapshot = homes(&mut emu);
        let value = rax_home(&snapshot);
        let mut value_sets = Vec::new();
        match place {
            "memory" => emu
                .state
                .store_concrete(&emu.arena, EXISTING_B, value, Width::W64),
            "register" => emu.state.set_reg(Reg::Rax, value),
            "flag" => {
                let flag = emu.arena.trunc(value, Width::W8);
                emu.state.set_flag(Flag::Cf, flag);
            }
            "xmm" => emu.state.set_xmm(0, XmmHalf::Low, value),
            "pin" => emu.pins.push((value, 7)),
            "value_set" => value_sets.push((value, 7)),
            "symbolic_store" => {
                let address = emu.arena.init_reg(Reg::R12);
                emu.state
                    .store_symbolic(&emu.arena, address, value, Width::W64);
            }
            _ => unreachable!(),
        }
        let error = rejection(LocationHomePlan::discover(&mut emu, &snapshot, &value_sets));
        assert!(
            error.contains("no cross-edge identity") || error.contains("symbolic-memory history"),
            "{place}: {error}"
        );
    }
}

#[test]
fn late_incoming_pure_state_and_metadata_changes_are_not_ignored() {
    let pe = fixture(None);
    for place in [
        "memory",
        "register",
        "flag",
        "xmm",
        "pin",
        "value_set",
        "written_arg",
        "vip",
        "unwind",
    ] {
        let mut first = incoming(&pe, 0, 0);
        let snapshot = homes(&mut first);
        let mut plan = LocationHomePlan::discover(&mut first, &snapshot, &[]).unwrap();
        let mut changed = first.clone();
        let one = changed.arena.constant(1, Width::W64);
        let mut value_sets = Vec::new();
        match place {
            "memory" => changed
                .state
                .store_concrete(&changed.arena, EXISTING_B, one, Width::W64),
            "register" => changed.state.set_reg(Reg::Rax, one),
            "flag" => {
                let flag = changed.arena.constant(1, Width::W8);
                changed.state.set_flag(Flag::Cf, flag);
            }
            "xmm" => changed.state.set_xmm(0, XmmHalf::Low, one),
            "pin" => changed.pins.push((one, 1)),
            "value_set" => value_sets.push((one, 1)),
            "written_arg" => {
                assert!(!changed.written_args.is_empty());
                changed.written_args.clear();
            }
            "vip" => changed.vip_slot = Some(EXISTING_B),
            "unwind" => changed.unwind_slot = Some(EXISTING_B),
            _ => unreachable!(),
        }
        assert!(
            plan.validate(&mut changed, &snapshot, &value_sets).is_err(),
            "{place} escaped late incoming validation"
        );
    }
}

#[test]
fn later_edge_requires_every_planned_byte_and_its_same_register_home() {
    let pe = fixture(None);
    let mut first = incoming(&pe, 0, 0);
    let snapshot = homes(&mut first);
    let mut plan = LocationHomePlan::discover(&mut first, &snapshot, &[]).unwrap();
    let address = LAYOUTS[0].slot(ACTIVE, Reg::Rax).unwrap();
    assert!(
        plan.locations
            .iter()
            .any(|location| location.address == address && location.home == Reg::Rax)
    );
    let mut missing = first.clone();
    missing.state.mem.remove(&(address + 3));
    assert!(
        plan.validate(&mut missing, &snapshot, &[])
            .unwrap_err()
            .contains("missing, partial")
    );
    let mut wrong = first.clone();
    let other = snapshot.iter().find(|(reg, _)| *reg == Reg::Rcx).unwrap().1;
    wrong
        .state
        .store_concrete(&wrong.arena, address, other, Width::W64);
    assert!(
        plan.validate(&mut wrong, &snapshot, &[])
            .unwrap_err()
            .contains("does not equal parent exit rax")
    );
}

#[test]
fn identical_values_with_different_call_provenance_reject_late_edge() {
    let pe = fixture(None);
    let mut first = incoming(&pe, 0, 0);
    let snapshot = homes(&mut first);
    let mut plan = LocationHomePlan::discover(&mut first, &snapshot, &[]).unwrap();
    let mut missing = first.clone();
    assert!(missing.guest_call_results.remove(&rax_home(&snapshot)));
    let error = plan.validate(&mut missing, &snapshot, &[]).unwrap_err();
    assert!(error.contains("call-result provenance"), "{error}");
}

#[test]
fn output_slot_sets_are_compared_on_every_incoming_edge() {
    let pe = fixture(None);
    let mut first = incoming(&pe, 0, 0);
    first.guest_call_output_slots.insert(STACK + 0x20);
    let snapshot = homes(&mut first);
    let mut plan = LocationHomePlan::discover(&mut first, &snapshot, &[]).unwrap();
    for slot in [None, Some(STACK + 0x21)] {
        let mut changed = first.clone();
        changed.guest_call_output_slots.clear();
        changed.guest_call_output_slots.extend(slot);
        let error = plan.validate(&mut changed, &snapshot, &[]).unwrap_err();
        assert!(error.contains("call-output slots"), "{error}");
    }
}

#[test]
fn live_call_source_inside_a_different_home_expression_is_not_guessed() {
    let pe = fixture(None);
    let mut emu = incoming(&pe, 0, 0);
    let snapshot = homes(&mut emu);
    let plan = LocationHomePlan::discover(&mut emu, &snapshot, &[]).unwrap();
    let five = emu.arena.constant(5, Width::W64);
    let displaced = emu.arena.bin(BinOp::Add, rax_home(&snapshot), five);
    for location in plan.locations.iter().filter(|location| location.home == Reg::Rax) {
        emu.state.store_concrete(&emu.arena, location.address, displaced, Width::W64);
    }
    let changed = homes(&mut emu);
    let error = rejection(LocationHomePlan::discover(&mut emu, &changed, &[]));
    assert!(error.contains("live call-result provenance has no seeded home"), "{error}");
}

#[test]
fn seeding_invalidates_partial_output_slot_overlap_but_preserves_disjoint_slot() {
    let pe = fixture(None);
    let mut emu = incoming(&pe, 0, 0);
    let address = LAYOUTS[1].slot(ALT, Reg::Rax).unwrap();
    emu.guest_call_output_slots.extend([address - 7, address + 7, STACK + 0x20]);
    let snapshot = homes(&mut emu);
    let plan = LocationHomePlan::discover(&mut emu, &snapshot, &[]).unwrap();
    plan.seed(&mut emu, BlockRef(91));
    assert_eq!(emu.guest_call_output_slots, HashSet::from([STACK + 0x20]));
}

#[test]
fn call_home_provenance_survives_clone_and_arena_compaction() {
    let pe = fixture(None);
    let mut first = incoming(&pe, 0, 19);
    let first_homes = homes(&mut first);
    let mut plan = LocationHomePlan::discover(&mut first, &first_homes, &[]).unwrap();
    let mut compacted = first.clone();
    compacted.compact_arena(&mut []);
    let compacted_homes = homes(&mut compacted);
    assert_ne!(rax_home(&first_homes), rax_home(&compacted_homes));
    plan.validate(&mut compacted, &compacted_homes, &[]).unwrap();
    plan.seed(&mut compacted, BlockRef(92));
    let result = read_rax(&mut compacted, READ_CANONICAL);
    assert!(compacted.guest_call_results.contains(&result));
}

#[test]
fn historical_event_only_call_sources_are_dropped_after_seeding() {
    let pe = fixture(Some(0));
    let mut emu = incoming(&pe, 0, 0);
    assert!(!emu.guest_call_results.is_empty());
    let snapshot = homes(&mut emu);
    let plan = LocationHomePlan::discover(&mut emu, &snapshot, &[]).unwrap();
    plan.seed(&mut emu, BlockRef(93));
    assert!(emu.guest_call_results.is_empty());
}

#[test]
fn a_call_source_still_used_outside_seeded_memory_rejects_the_join() {
    let pe = fixture(None);
    let mut emu = incoming(&pe, 0, 0);
    let snapshot = homes(&mut emu);
    let plan = LocationHomePlan::discover(&mut emu, &snapshot, &[]).unwrap();
    let value = emu.arena.init_reg(Reg::R15);
    for location in plan.locations.iter().filter(|location| location.home == Reg::Rax) {
        emu.state.store_concrete(&emu.arena, location.address, value, Width::W64);
    }
    emu.state.set_reg(Reg::R15, value);
    emu.guest_call_results.clear();
    emu.guest_call_results.insert(value);
    let changed = homes(&mut emu);
    let error = rejection(LocationHomePlan::discover(&mut emu, &changed, &[]));
    assert!(error.contains("call-result provenance remains in unseeded state"), "{error}");
}

#[test]
fn a_shared_call_origin_is_not_equal_to_two_independent_call_origins() {
    let pe = fixture(None);
    let mut first = incoming(&pe, 0, 0);
    let snapshot = homes(&mut first);
    let initial_plan = LocationHomePlan::discover(&mut first, &snapshot, &[]).unwrap();
    let source = rax_home(&snapshot);
    for location in initial_plan.locations.iter().filter(|location| location.home == Reg::Rcx) {
        first.state.store_concrete(&first.arena, location.address, source, Width::W64);
    }
    let first_homes = homes(&mut first);
    let mut plan = LocationHomePlan::discover(&mut first, &first_homes, &[]).unwrap();
    let mut second = first.clone();
    let other = second.arena.opaque("independent_call_result", Width::W64);
    second.guest_call_results.insert(other);
    for location in plan.locations.iter().filter(|location| location.home == Reg::Rcx) {
        second.state.store_concrete(&second.arena, location.address, other, Width::W64);
    }
    let second_homes = homes(&mut second);
    let error = plan.validate(&mut second, &second_homes, &[]).unwrap_err();
    assert!(error.contains("call-result provenance"), "{error}");
}

#[test]
fn seeding_a_derived_pointer_does_not_lose_its_source_in_another_home() {
    let pe = fixture(None);
    let mut emu = incoming(&pe, 0, 0);
    let snapshot = homes(&mut emu);
    let plan = LocationHomePlan::discover(&mut emu, &snapshot, &[]).unwrap();
    let five = emu.arena.constant(5, Width::W64);
    let field = emu.arena.bin(BinOp::Add, rax_home(&snapshot), five);
    assert!(emu.is_guest_call_address(field));
    for location in plan.locations.iter().filter(|location| location.home == Reg::Rcx) {
        emu.state.store_concrete(&emu.arena, location.address, field, Width::W64);
    }
    let changed = homes(&mut emu);
    let error = rejection(LocationHomePlan::discover(&mut emu, &changed, &[]));
    assert!(error.contains("call-result provenance has no seeded home for a nested use"), "{error}");
}

#[test]
fn deep_live_call_provenance_exhaustion_rejects_instead_of_dropping_the_source() {
    let pe = fixture(None);
    let mut emu = incoming(&pe, 0, 0);
    let snapshot = homes(&mut emu);
    let plan = LocationHomePlan::discover(&mut emu, &snapshot, &[]).unwrap();
    let mut value = rax_home(&snapshot);
    for _ in 0..=CALL_PROVENANCE_NODE_BUDGET {
        value = emu.arena.load(value, Width::W64);
    }
    for location in plan.locations.iter().filter(|location| location.home == Reg::Rax) {
        emu.state.store_concrete(&emu.arena, location.address, value, Width::W64);
    }
    let changed = homes(&mut emu);
    let error = rejection(LocationHomePlan::discover(&mut emu, &changed, &[]));
    assert!(error.contains("call-provenance proof budget exhausted"), "{error}");
}
