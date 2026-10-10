//! Guest call results may address observable memory even without entry-register roots.
use super::*;
use crate::binary::pe::Section;
use crate::ir::sched::{self, Instr};
use iced_x86::code_asm::*;

const BASE: u64 = 0x180000000;
const START: u64 = BASE + 0x1000;
const HELPER: u64 = BASE + 0x3000;
const STACK: u64 = crate::vm::DEFAULT_STACK_BASE;

fn fixture(build: impl FnOnce(&mut CodeAssembler)) -> PeFile {
    let mut a = CodeAssembler::new(64).unwrap();
    build(&mut a);
    let code = a.assemble(START).unwrap();
    let mut data = vec![0x90; 0x1300];
    data[0x200..0x200 + code.len()].copy_from_slice(&code);
    data[0x1200] = 0xc3;
    PeFile {
        data, image_base: BASE, entry_point_rva: 0x1000, size_of_image: 0x4000,
        sections: vec![
            Section { name: ".tvm0".into(), virtual_address: 0x1000, virtual_size: 0x1000,
                raw_address: 0x200, raw_size: 0x1000, characteristics: 0x60000020 },
            Section { name: ".text".into(), virtual_address: 0x3000, virtual_size: 0x100,
                raw_address: 0x1200, raw_size: 0x100, characteristics: 0x60000020 },
        ], opt_header_offset: 0, section_table_offset: 0, file_alignment: 0x200,
        section_alignment: 0x1000, size_of_headers: 0x200, loader_bound: vec![],
    }
}

#[test]
fn callret_memory_retains_direct_offset_and_loaded_pointer_effects() {
    let pe = fixture(|a| {
        a.call(HELPER).unwrap();
        a.mov(qword_ptr(rax), 42).unwrap();
        a.mov(qword_ptr(rax + 0x18), 43).unwrap();
        a.mov(rcx, qword_ptr(rax + 8)).unwrap();
        a.mov(qword_ptr(rcx + 0x10), 44).unwrap();
        a.mov(eax, 7).unwrap();
        a.ret().unwrap();
    });
    let mut emu = Emulator::new(&pe, STACK);
    assert!(matches!(emu.run(START, 100), Stop::Return { .. }));
    let stores: Vec<_> = emu.events.iter().filter_map(|e| match e {
        Event::Store { region, value, .. } if emu.arena.as_const(*value).is_some_and(|v| (42..=44).contains(&v)) => Some(*region),
        _ => None,
    }).collect();
    assert_eq!(stores, vec![Region::Guest; 3]);
    assert!(emu.events.iter().any(|e| matches!(e, Event::Load { region: Region::Guest, addr, .. }
        if emu.arena.as_const(*addr).is_none())));
    let cfg = crate::vm::explore::Explorer::new(&pe, STACK).recover(START);
    let blocks = sched::schedule(&cfg);
    let stores: Vec<_> = blocks.iter().flat_map(|b| b.instrs.iter()).filter_map(|i| match i {
        Instr::Store { value: sched::Operand::Imm(v), .. } if (42..=44).contains(v) => Some(*v),
        _ => None,
    }).collect();
    assert_eq!(stores, vec![42, 43, 44], "the heap writes must survive lowering");
}

#[test]
fn callret_memory_proof_survives_clone_event_drain_and_compaction() {
    let pe = fixture(|a| { a.call(HELPER).unwrap(); a.nop().unwrap(); a.ret().unwrap(); });
    let mut emu = Emulator::new(&pe, STACK);
    assert!(matches!(emu.run(START, 1), Stop::Budget { .. }));
    let mut cloned = emu.clone();
    cloned.events.clear();
    for _ in 0..100 { cloned.arena.opaque("dead", Width::W64); }
    cloned.compact_arena(&mut []);
    let result = cloned.state.reg(Reg::Rax);
    assert_eq!(cloned.region_of_symbolic(result), Region::Guest);
    for name in ["call_ret", "clobbered", "boxed", "unknown"] {
        let unproved = cloned.arena.opaque(name, Width::W64);
        assert_eq!(cloned.region_of_symbolic(unproved), Region::VmInternal,
            "an opaque label alone is not a guest-call result proof");
    }
    let vm = cloned.arena.constant(START, Width::W64);
    let dispatch = cloned.arena.bin(BinOp::Add, vm, result);
    assert_eq!(cloned.region_of_symbolic(dispatch), Region::VmInternal);
}

#[test]
fn callret_memory_guest_calls_separate_heap_and_global_load_generations() {
    let pe = fixture(|a| { a.call(HELPER).unwrap(); a.nop().unwrap(); a.ret().unwrap(); });
    let mut emu = Emulator::new(&pe, STACK);
    let heap = emu.arena.init_reg(Reg::Rbx);
    let global = emu.arena.constant(BASE + 0x5000, Width::W64);
    let before_heap = emu.load(heap, Width::W64, START);
    let before_global = emu.load(global, Width::W64, START);
    assert!(matches!(emu.run(START, 1), Stop::Budget { .. }));
    let after_heap = emu.load(heap, Width::W64, START + 5);
    let after_global = emu.load(global, Width::W64, START + 5);
    assert_ne!(before_heap, after_heap, "external calls may change heap memory");
    assert_ne!(before_global, after_global, "external calls may change concrete globals");
    assert_eq!(emu.load(heap, Width::W64, START + 5), after_heap);
    assert_eq!(emu.load(global, Width::W64, START + 5), after_global);
}

#[test]
fn callret_memory_field_offsets_do_not_become_vm_stack_bases() {
    let pe = fixture(|a| { a.call(HELPER).unwrap(); a.nop().unwrap(); a.ret().unwrap(); });
    let mut emu = Emulator::new(&pe, STACK);
    assert!(matches!(emu.run(START, 1), Stop::Budget { .. }));
    let result = emu.state.reg(Reg::Rax);
    emu.guest_ctx = Some(STACK - 0x500);
    emu.guest_layout = Some(LAYOUTS[0]);
    for displacement in [8u64, 0x18, u64::MAX - 7] {
        let offset = emu.arena.constant(displacement, Width::W64);
        let address = emu.arena.bin(BinOp::Add, result, offset);
        assert_eq!(emu.region_of_symbolic(address), Region::Guest);
        let value = emu.arena.constant(0x55, Width::W64);
        emu.store(address, value, Width::W64, START + 5);
        assert!(matches!(emu.events.last(), Some(Event::Store { region: Region::Guest, .. })));
    }
    for base in [START, STACK - 0x1000] {
        let base = emu.arena.constant(base, Width::W64);
        let address = emu.arena.bin(BinOp::Add, result, base);
        assert_eq!(emu.region_of_symbolic(address), Region::VmInternal);
    }
    let mask = emu.arena.constant(7, Width::W64);
    let index = emu.arena.bin(BinOp::And, result, mask);
    assert!(!emu.is_guest_call_address(index), "a scalar index is not a pointer base");
}

#[test]
fn call_output_pointer_load_retains_object_stores() {
    let pe = fixture(|a| {
        a.sub(rsp, 0x48).unwrap();
        a.mov(qword_ptr(rsp + 0x20), 0).unwrap();
        a.lea(rcx, qword_ptr(rsp + 0x20)).unwrap();
        a.call(HELPER).unwrap();
        a.mov(rax, qword_ptr(rsp + 0x20)).unwrap();
        a.mov(word_ptr(rax + 5), 0xd3).unwrap();
        a.mov(dword_ptr(rax + 0x5f), 1).unwrap();
        a.add(rsp, 0x48).unwrap();
        a.ret().unwrap();
    });
    let mut emu = Emulator::new(&pe, STACK);
    assert!(matches!(emu.run(START, 100), Stop::Return { .. }));
    let output = emu.state.reg(Reg::Rax);
    assert!(matches!(emu.arena.op(output), Op::Load(_, _)),
        "a call output must not reuse the pre-call zero initialization");
    let writes: Vec<_> = emu.events.iter().filter_map(|e| match e {
        Event::Store { addr, width, region, .. } if emu.arena.as_const(*addr).is_none() => Some((*width,*region)),
        _ => None,
    }).collect();
    assert_eq!(writes, vec![(Width::W16,Region::Guest),(Width::W32,Region::Guest)]);
}

#[test]
fn unpassed_stack_slot_and_overwritten_output_are_not_call_outputs() {
    let pe = fixture(|a| { a.lea(rcx, qword_ptr(rsp + 0x20)).unwrap(); a.call(HELPER).unwrap(); a.ret().unwrap(); });
    let mut emu = Emulator::new(&pe, STACK);
    assert!(matches!(emu.run(START, 2), Stop::Budget { .. }));
    let passed = emu.arena.constant(STACK + 0x20, Width::W64);
    let unpassed = emu.arena.constant(STACK + 0x28, Width::W64);
    let yes = emu.load(passed, Width::W64, START);
    let no = emu.load(unpassed, Width::W64, START);
    assert_eq!(emu.region_of_symbolic(yes), Region::Guest);
    assert_eq!(emu.region_of_symbolic(no), Region::VmInternal);
    let mut cloned = emu.clone();
    cloned.events.clear();
    cloned.state.set_reg(Reg::Rbx, yes);
    cloned.compact_arena(&mut []);
    assert_eq!(cloned.region_of_symbolic(cloned.state.reg(Reg::Rbx)), Region::Guest);
    let overlapping = emu.arena.constant(STACK + 0x24, Width::W64);
    emu.forget_memory(overlapping, 1);
    let overwritten = emu.load(passed, Width::W64, START);
    assert_eq!(emu.region_of_symbolic(overwritten), Region::VmInternal);
}

#[test]
fn vm_private_stack_pointer_is_not_an_external_call_output() {
    let pe = fixture(|a| { a.ret().unwrap(); });
    let mut emu = Emulator::new(&pe, STACK);
    emu.guest_ctx = Some(STACK - 0x500);
    emu.guest_layout = Some(LAYOUTS[0]);
    let context = emu.arena.constant(STACK - 0x400, Width::W64);
    emu.expose_call_stack_arguments(&[(Reg::Rcx, context)]);
    assert!(emu.guest_call_output_slots.is_empty());
}

#[test]
fn call_output_invalidates_both_sides_and_unbounded_caller_extent() {
    let pe = fixture(|a| { a.ret().unwrap(); });
    let mut emu = Emulator::new(&pe, STACK);
    let value = emu.arena.constant(0x1234, Width::W64);
    for address in [STACK + 0x18, STACK + 0x20,
        STACK + crate::ir::sched::FRAME_WINDOW + 8] {
        emu.state.store_concrete(&emu.arena, address, value, Width::W64);
    }
    let slot = emu.arena.constant(STACK + 0x20, Width::W64);
    emu.expose_call_stack_arguments(&[(Reg::Rcx, slot)]);
    assert!(emu.state.mem.is_empty(), "an output pointer does not bound the callee's write extent");
}

#[test]
fn call_output_uses_current_call_rsp_not_stale_context_fallback() {
    let pe = fixture(|a| { a.ret().unwrap(); });
    let mut emu = Emulator::new(&pe, STACK);
    emu.guest_ctx = Some(STACK - 0x500);
    emu.guest_layout = Some(LAYOUTS[0]);
    let actual_rsp = emu.arena.constant(STACK - 0x100, Width::W64);
    emu.state.set_reg(Reg::Rsp, actual_rsp);
    let slot = emu.arena.constant(STACK - 0xe0, Width::W64);
    let old = emu.arena.constant(0x1234, Width::W64);
    emu.state.store_concrete(&emu.arena, STACK - 0xe8, old, Width::W64);
    assert_eq!(emu.guest_stack_floor(), STACK, "fixture exercises missing/stale image RSP");
    emu.expose_call_stack_arguments(&[(Reg::Rcx, slot)]);
    assert!(emu.guest_call_output_slots.contains(&(STACK - 0xe0)));
    assert!(!emu.state.mem.contains_key(&(STACK - 0xe8)));
}
