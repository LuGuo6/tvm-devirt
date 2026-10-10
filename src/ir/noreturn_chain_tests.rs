//! Straight-line native wrapper proofs; nested bodies remain original calls.
use super::*;
use crate::binary::pe::Section;
use iced_x86::code_asm::*;

const BASE: u64 = 0x180000000;
const CALLER: u64 = BASE + 0x1000;
const WRAPPER: u64 = BASE + 0x2000;
const HELPER: u64 = BASE + 0x2700;
const SECOND: u64 = BASE + 0x2800;

fn fixture(wrapper: &[u8]) -> PeFile {
    let mut data = vec![0x90; 0x2300];
    data[0x1200..0x1200 + wrapper.len()].copy_from_slice(wrapper);
    data[0x1900] = 0xc3;
    data[0x1a00] = 0xc3;
    let mut caller = CodeAssembler::new(64).unwrap();
    caller.mov(ecx, 5).unwrap();
    caller.call(WRAPPER).unwrap();
    caller.nop().unwrap();
    caller.ret().unwrap();
    let caller = caller.assemble(CALLER).unwrap();
    data[0x200..0x200 + caller.len()].copy_from_slice(&caller);
    PeFile {
        data,
        image_base: BASE,
        entry_point_rva: 0x1000,
        size_of_image: 0x4000,
        sections: vec![
            Section {
                name: ".tvm0".into(),
                virtual_address: 0x1000,
                virtual_size: 0x1000,
                raw_address: 0x200,
                raw_size: 0x1000,
                characteristics: 0x60000020,
            },
            Section {
                name: ".text".into(),
                virtual_address: 0x2000,
                virtual_size: 0x1000,
                raw_address: 0x1200,
                raw_size: 0x1000,
                characteristics: 0x60000020,
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

fn assemble(build: impl FnOnce(&mut CodeAssembler)) -> Vec<u8> {
    let mut a = CodeAssembler::new(64).unwrap();
    build(&mut a);
    a.assemble(WRAPPER).unwrap()
}

#[test]
fn noreturn_chain_accepts_multiple_calls_and_preserves_complete_outer_call_event() {
    for count in [2, 3] {
        let code = assemble(|a| {
            a.sub(rsp, 0x48).unwrap();
            for index in 0..count {
                a.lea(rcx, ptr(rsp + 0x20)).unwrap();
                a.call(if index % 2 == 0 { HELPER } else { SECOND })
                    .unwrap();
            }
            a.int3().unwrap();
        });
        let pe = fixture(&code);
        assert!(callee_never_returns(&pe, WRAPPER));
        let original_wrapper = pe.read_va(WRAPPER, code.len()).unwrap().to_vec();
        let mut emu = Emulator::new(&pe, crate::vm::DEFAULT_STACK_BASE);
        let stop = emu.run(CALLER, 30);
        let Stop::NoReturn { site } = stop else {
            panic!("{stop:?}")
        };
        assert_eq!(site, CALLER + 5);
        let calls: Vec<_> = emu
            .events
            .iter()
            .filter_map(|e| match e {
                Event::Call {
                    target,
                    site,
                    args,
                    ret,
                    ..
                } => Some((target, site, args, ret)),
                _ => None,
            })
            .collect();
        assert_eq!(
            calls.len(),
            1,
            "retain outer native call; never inline/discard nested calls"
        );
        let (target, site, args, ret) = calls[0];
        assert_eq!(emu.arena.as_const(target.unwrap()), Some(WRAPPER));
        assert_eq!(*site, CALLER + 5);
        assert!(
            args.iter()
                .any(|(reg, value)| *reg == Reg::Rcx && emu.arena.as_const(*value) == Some(5))
        );
        assert!(
            ret.is_some(),
            "preserve the existing call event representation"
        );
        assert_eq!(pe.read_va(WRAPPER, code.len()).unwrap(), original_wrapper);
    }
}

#[test]
fn noreturn_chain_returning_or_nested_throw_paths_do_not_authorize_the_wrapper() {
    for nested_trap in [false, true] {
        let code = assemble(|a| {
            a.call(HELPER).unwrap();
            a.call(SECOND).unwrap();
            a.ret().unwrap();
            a.int3().unwrap();
        });
        let mut pe = fixture(&code);
        if nested_trap {
            pe.data[0x1900] = 0xcc;
        }
        assert!(!callee_never_returns(&pe, WRAPPER));
        let mut emu = Emulator::new(&pe, crate::vm::DEFAULT_STACK_BASE);
        assert!(matches!(emu.run(CALLER, 30), Stop::Return { .. }));
    }
}

#[test]
fn noreturn_chain_rejects_return_branches_indirect_transfers_and_cycles() {
    let codes = [
        assemble(|a| {
            a.test(ecx, ecx).unwrap();
            a.je(WRAPPER + 0x40).unwrap();
            a.call(HELPER).unwrap();
            a.int3().unwrap();
        }),
        assemble(|a| {
            a.call(rax).unwrap();
            a.int3().unwrap();
        }),
        assemble(|a| {
            a.call(HELPER).unwrap();
            a.jmp(rax).unwrap();
            a.int3().unwrap();
        }),
        assemble(|a| {
            a.call(HELPER).unwrap();
            a.jmp(WRAPPER).unwrap();
            a.int3().unwrap();
        }),
        assemble(|a| {
            a.call(WRAPPER).unwrap();
            a.int3().unwrap();
        }),
        vec![0xe2, 0xfe, 0xcc], // LOOP backedge is not a common Jcc.
        vec![0xe3, 0x3e, 0xcc], // JRCXZ with a reachable RET branch.
        vec![0x0f, 0x05, 0xcc], // SYSCALL is not a direct native CALL.
        vec![0xf4, 0xcc],       // HLT has no ordinary fallthrough contract here.
        vec![0xcb, 0xcc],       // RETF is a return even with trailing trap padding.
    ];
    for code in codes {
        let mut pe = fixture(&code);
        pe.data[0x1240] = 0xc3;
        assert!(!callee_never_returns(&pe, WRAPPER), "accepted {code:02x?}");
    }
}

#[test]
fn noreturn_chain_budget_includes_the_terminal_trap() {
    for (nops, expected) in [
        (NORETURN_SCAN_INSTRS - 1, true),
        (NORETURN_SCAN_INSTRS, false),
    ] {
        let mut code = vec![0x90; nops];
        code.push(0xcc);
        assert_eq!(callee_never_returns(&fixture(&code), WRAPPER), expected);
    }
}

#[test]
fn noreturn_chain_requires_complete_file_backed_instructions_in_one_executable_section() {
    let code = assemble(|a| {
        a.call(HELPER).unwrap();
        a.call(SECOND).unwrap();
        a.ud2().unwrap();
    });
    let mut pe = fixture(&code);
    assert!(callee_never_returns(&pe, WRAPPER));
    pe.sections[1].characteristics &= !0x20000000;
    assert!(!callee_never_returns(&pe, WRAPPER));
    let mut pe = fixture(&code);
    pe.sections[1].raw_size = code.len() as u32 - 1;
    assert!(
        !callee_never_returns(&pe, WRAPPER),
        "truncated UD2 must not read across raw end"
    );
    let mut pe = fixture(&[0x90, 0xcc]);
    pe.sections[1].raw_size = 1;
    pe.sections[1].virtual_size = 1;
    pe.sections.push(Section {
        name: ".adjacent".into(),
        virtual_address: 0x2001,
        virtual_size: 0x100,
        raw_address: 0x1201,
        raw_size: 0x100,
        characteristics: 0x60000020,
    });
    assert!(
        !callee_never_returns(&pe, WRAPPER),
        "adjacent section trap is outside the proof"
    );
}

#[test]
fn noreturn_chain_rejects_unmapped_call_targets_and_invalid_bytes() {
    let code = assemble(|a| {
        a.call(BASE + 0x5000).unwrap();
        a.call(HELPER).unwrap();
        a.int3().unwrap();
    });
    assert!(!callee_never_returns(&fixture(&code), WRAPPER));
    assert!(!callee_never_returns(
        &fixture(&[0x0f, 0xff, 0xcc]),
        WRAPPER
    ));
}
