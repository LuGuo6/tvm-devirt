use super::*;
use crate::binary::pe::Section;
use crate::ir::expr::BlockRef;
use iced_x86::code_asm::*;

const IMAGE: u64 = 0x180000000;
const START: u64 = IMAGE + 0x1000;
const HELPER: u64 = IMAGE + 0x4000;
const STACK: u64 = 0x7fff00003000;
const ALT: u64 = STACK - 0x300;
const ACTIVE: u64 = STACK - 0x100;
const STALE: u64 = STACK - 0x600;

fn registers() -> [(Reg, AsmRegister64); 16] {
    [
        (Reg::Rax, rax),
        (Reg::Rbx, rbx),
        (Reg::Rcx, rcx),
        (Reg::Rdx, rdx),
        (Reg::Rsp, rsp),
        (Reg::Rbp, rbp),
        (Reg::Rsi, rsi),
        (Reg::Rdi, rdi),
        (Reg::R8, r8),
        (Reg::R9, r9),
        (Reg::R10, r10),
        (Reg::R11, r11),
        (Reg::R12, r12),
        (Reg::R13, r13),
        (Reg::R14, r14),
        (Reg::R15, r15),
    ]
}

pub(crate) fn round_trip() -> PeFile {
    round_trip_with(None, false)
}

fn round_trip_with(omit_copy: Option<Reg>, stale_consumer: bool) -> PeFile {
    let mut a = CodeAssembler::new(64).unwrap();
    for (reg, machine) in registers() {
        if matches!(reg, Reg::R10 | Reg::Rsp) {
            continue;
        }
        let offset = LAYOUTS[0].slot(0, reg).unwrap() as i32;
        // Both NOTs must cancel in the storage-origin proof.
        a.mov(machine, qword_ptr(r10 + offset)).unwrap();
        a.not(machine).unwrap();
        a.not(machine).unwrap();
    }
    a.push(qword_ptr(r10 + 0x20)).unwrap();
    a.mov(r10, qword_ptr(r10 + 0x50)).unwrap();
    a.mov(rsp, qword_ptr(rsp)).unwrap();
    a.call(HELPER).unwrap();
    a.lea(rsp, ptr(rsp - 0x300)).unwrap();
    let old_rbp = LAYOUTS[1].slot(0, Reg::Rbp).unwrap() as i32;
    a.mov(qword_ptr(rsp + old_rbp), rbp).unwrap();
    a.mov(rbp, rsp).unwrap();
    for (reg, machine) in registers() {
        if reg == Reg::Rbp {
            continue;
        }
        let offset = LAYOUTS[1].slot(0, reg).unwrap() as i32;
        a.mov(qword_ptr(rbp + offset), machine).unwrap();
    }
    let rsp_slot = LAYOUTS[1].slot(0, Reg::Rsp).unwrap() as i32;
    a.add(qword_ptr(rbp + rsp_slot), 0x300).unwrap();
    a.lea(r10, ptr(rbp + 0x200)).unwrap();
    for (reg, _) in registers() {
        if Some(reg) == omit_copy {
            continue;
        }
        let from = LAYOUTS[1].slot(0, reg).unwrap() as i32;
        let to = LAYOUTS[0].slot(0, reg).unwrap() as i32;
        a.push(qword_ptr(rbp + from)).unwrap();
        a.pop(qword_ptr(r10 + to)).unwrap();
    }
    if stale_consumer {
        // B already exists. Switching the live pointer needs no new complete
        // copy, and A's round-trip candidate cannot prove which one is read.
        a.mov(r8, STALE).unwrap();
        a.mov(rax, qword_ptr(r8 + 0x38)).unwrap();
    }
    a.int3().unwrap();
    let code = a.assemble(START).unwrap();
    let mut data = vec![0u8; 0x2300];
    data[0x200..0x200 + code.len()].copy_from_slice(&code);
    data[0x2200] = 0xc3;
    PeFile {
        data,
        image_base: IMAGE,
        entry_point_rva: 0x1000,
        size_of_image: 0x5000,
        sections: vec![
            Section {
                name: ".tvm0".into(),
                virtual_address: 0x1000,
                virtual_size: 0x2000,
                raw_address: 0x200,
                raw_size: 0x2000,
                characteristics: 0x20000000,
            },
            Section {
                name: ".text".into(),
                virtual_address: 0x4000,
                virtual_size: 0x100,
                raw_address: 0x2200,
                raw_size: 0x100,
                characteristics: 0x20000000,
            },
        ],
        opt_header_offset: 0,
        section_table_offset: 0,
        file_alignment: 0x200,
        section_alignment: 0x1000,
        size_of_headers: 0x200,
        loader_bound: vec![],
    }
}

pub(crate) fn prepared(pe: &PeFile) -> Emulator<'_> {
    let mut emu = Emulator::new(pe, STACK);
    emu.diagnostic_context_origin = true;
    for (reg, _) in registers() {
        let value = if reg == Reg::Rsp {
            emu.arena.constant(STACK, Width::W64)
        } else {
            emu.arena.init_reg(reg)
        };
        emu.state.store_concrete(
            &emu.arena,
            LAYOUTS[0].slot(ACTIVE, reg).unwrap(),
            value,
            Width::W64,
        );
    }
    let active = emu.arena.constant(ACTIVE, Width::W64);
    emu.state.set_reg(Reg::R10, active);
    // The former locator prefers this completed ALT save over an equally
    // matching live image. The complete round trip must supersede that guess.
    emu.guest_ctx = Some(ALT);
    emu.guest_layout = Some(LAYOUTS[1]);
    emu
}

pub(crate) fn prepared_locations(pe: &PeFile) -> Emulator<'_> {
    let mut emu = prepared(pe);
    emu.diagnostic_context_origin = false;
    emu.diagnostic_location_homes = true;
    emu
}

#[test]
fn context_round_trip_seeds_live_image_instead_of_the_old_spill_mirror() {
    let pe = round_trip();
    let mut emu = prepared(&pe);
    emu.run(START, 1000);
    assert_eq!(
        emu.guest_context_identity(),
        Ok(Some(ContextIdentity {
            base: ACTIVE,
            layout_index: 0,
            guest_rsp: STACK
        }))
    );
    let before = emu
        .state
        .load_concrete(
            &mut emu.arena,
            LAYOUTS[1].slot(ALT, Reg::Rdi).unwrap(),
            Width::W64,
            |_| None,
        )
        .unwrap();
    emu.seed_guest_params(BlockRef(9));
    let live = emu
        .state
        .load_concrete(
            &mut emu.arena,
            LAYOUTS[0].slot(ACTIVE, Reg::Rdi).unwrap(),
            Width::W64,
            |_| None,
        )
        .unwrap();
    let old = emu
        .state
        .load_concrete(
            &mut emu.arena,
            LAYOUTS[1].slot(ALT, Reg::Rdi).unwrap(),
            Width::W64,
            |_| None,
        )
        .unwrap();
    assert_eq!(emu.arena.op(live), &Op::Param(BlockRef(9), Reg::Rdi));
    assert_eq!(
        old, before,
        "seeding must update only the diagnostic round-trip candidate in this fixture"
    );
    // Parameterized GPRs change content scores. The diagnostic candidate's
    // descriptor/layout must survive without reverting to the old mirror.
    let regs = emu.guest_registers().unwrap();
    assert!(
        regs.iter()
            .any(|(r, v)| *r == Reg::Rdi && emu.arena.op(*v) == &Op::Param(BlockRef(9), Reg::Rdi))
    );
    assert_eq!(
        emu.guest_context_identity().unwrap().unwrap().guest_rsp,
        STACK
    );
}

#[test]
fn context_incomplete_return_copy_never_falls_back_to_a_cached_image() {
    let pe = round_trip_with(Some(Reg::Rdi), false);
    let mut emu = prepared(&pe);
    emu.run(START, 1000);
    assert_eq!(emu.guest_context_identity(), Err(()));
    assert!(emu.seed_guest_params(BlockRef(9)).is_empty());
    assert!(emu.guest_registers().is_none());
}

#[test]
fn context_round_trip_rejects_a_guest_frame_outside_the_scheduler_window() {
    let pe = round_trip();
    let mut emu = prepared(&pe);
    emu.run(START, 1000);
    let far = emu
        .arena
        .constant(STACK + crate::ir::sched::FRAME_WINDOW + 8, Width::W64);
    emu.state.store_concrete(
        &emu.arena,
        LAYOUTS[0].slot(ACTIVE, Reg::Rsp).unwrap(),
        far,
        Width::W64,
    );
    assert_eq!(emu.guest_context_identity(), Err(()));
}

#[test]
fn context_round_trip_is_not_a_proof_of_the_actual_next_consumer() {
    let pe = round_trip_with(None, true);
    let mut emu = prepared(&pe);
    let sentinel = emu.arena.constant(0x1020_3040_5060_7080, Width::W64);
    emu.state.store_concrete(
        &emu.arena,
        LAYOUTS[0].slot(STALE, Reg::Rdi).unwrap(),
        sentinel,
        Width::W64,
    );
    emu.run(START, 1000);
    assert_eq!(
        emu.arena.as_const(emu.state.reg(Reg::Rax)),
        Some(0x1020_3040_5060_7080)
    );
    assert_eq!(
        emu.context_tracker.current(),
        Some((ACTIVE, 0)),
        "the candidate is A while the actual instruction read B"
    );
    // The production path rejects this experimental promotion: even a complete
    // candidate in the tracker cannot replace the pre-existing cached image.
    emu.diagnostic_context_origin = false;
    emu.guest_ctx = Some(ALT);
    emu.guest_layout = Some(LAYOUTS[1]);
    assert_eq!(emu.guest_context_identity(), Ok(None));
    assert_eq!(emu.locate_guest_context(), Some(ALT));
    let params = emu.seed_guest_params(BlockRef(9));
    assert!(
        params.iter().any(|(reg, _)| *reg == Reg::Rsp),
        "the diagnostic concrete-RSP exception must also be off"
    );
    let candidate_rdi = emu
        .raw_slot(LAYOUTS[0].slot(ACTIVE, Reg::Rdi).unwrap())
        .unwrap();
    assert!(!matches!(
        emu.arena.op(candidate_rdi),
        Op::Param(BlockRef(9), Reg::Rdi)
    ));
}

#[test]
fn context_spill_votes_cannot_combine_different_layouts_into_one_image() {
    let pe = round_trip();
    let mut emu = Emulator::new(&pe, STACK);
    emu.diagnostic_context_origin = false;
    for (layout, regs) in [
        (0, [Reg::Rax, Reg::Rbx, Reg::Rcx]),
        (1, [Reg::Rbp, Reg::Rdi, Reg::Rdx]),
    ] {
        for reg in regs {
            let addr = emu
                .arena
                .constant(LAYOUTS[layout].slot(ACTIVE, reg).unwrap(), Width::W64);
            let value = emu.arena.init_reg(reg);
            emu.events.push(Event::Store {
                addr,
                value,
                width: Width::W64,
                site: START,
                region: Region::Guest,
            });
        }
    }
    assert_eq!(
        emu.context_from_entry_spills(),
        None,
        "each layout has only three of the six required spills"
    );
}

#[test]
fn location_actual_implicit_and_unknown_writes_expire_storage_provenance() {
    // REP STOSQ has an implicit range, while MOV [RAX] has an unknown address.
    // Neither may retain the previously executed complete-save qualification.
    for code in [vec![0xf3, 0x48, 0xab, 0xcc], vec![0x48, 0x89, 0x18, 0xcc]] {
        let mut pe = round_trip();
        pe.data[0x1200..0x1200 + code.len()].copy_from_slice(&code);
        let mut emu = prepared(&pe);
        emu.diagnostic_context_origin = false;
        emu.diagnostic_location_homes = true;
        emu.run(START, 1000);
        assert!(!emu.context_tracker.saved_images().is_empty());
        let address = emu.arena.init_reg(Reg::Rax);
        let count = emu.arena.constant(2, Width::W64);
        emu.state.set_reg(Reg::Rax, address);
        emu.state.set_reg(Reg::Rdi, address);
        emu.state.set_reg(Reg::Rcx, count);
        emu.run(IMAGE + 0x2000, 100);
        assert!(emu.context_tracker.saved_images().is_empty());
    }
}
