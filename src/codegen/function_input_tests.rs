//! End-to-end fixtures: immutable function input differs from block-entry state.
use super::*;
use crate::binary::pe::Section;
use crate::vm::{DEFAULT_STACK_BASE, explore::Explorer};

const IMAGE: u64 = 0x180000000;
const START: u64 = IMAGE + 0x1000;
const HELPER: u64 = IMAGE + 0x3000;
const OUTPUT: u64 = IMAGE + 0x5000;

fn fixture(kind: &str) -> (PeFile, Vec<u8>) {
    let mut a = CodeAssembler::new(64).unwrap();
    let mut other = a.create_label();
    a.sub(rsp, 0x38).unwrap();
    if kind.starts_with("lock") {
        let offset = if kind == "lock_c0" { 0xc0 } else { 0x20 };
        a.lea(rcx, ptr(rcx + offset)).unwrap();
        a.test(edx, edx).unwrap();
        a.je(other).unwrap();
        a.mov(dword_ptr(rcx + 0x48), 0).unwrap();
        a.lea(rcx, ptr(rcx + 0x10)).unwrap();
        a.call(HELPER).unwrap();
    } else {
        // R9's input only survives in caller memory; no register is a valid
        // current-block substitute for it after R9 becomes scalar flags.
        a.mov(qword_ptr(rsp + 0x20), r9).unwrap();
        a.mov(r9d, 0x56).unwrap();
        a.test(ecx, ecx).unwrap();
        a.je(other).unwrap();
        a.mov(r10, qword_ptr(rsp + 0x20)).unwrap();
        a.inc(qword_ptr(r10)).unwrap();
    }
    a.mov(eax, 1).unwrap();
    a.add(rsp, 0x38).unwrap();
    a.ret().unwrap();
    a.set_label(&mut other).unwrap();
    a.mov(eax, 2).unwrap();
    a.add(rsp, 0x38).unwrap();
    a.ret().unwrap();
    let original = a.assemble(START).unwrap();
    let mut data = vec![0x90; 0x1300];
    data[0x200..0x200 + original.len()].copy_from_slice(&original);
    data[0x1200] = 0xc3;
    let pe = PeFile {
        data, image_base: IMAGE, entry_point_rva: 0x1000, size_of_image: 0x4000,
        sections: vec![
            Section { name: ".tvm0".into(), virtual_address: 0x1000, virtual_size: 0x1000,
                raw_address: 0x200, raw_size: 0x1000, characteristics: 0x60000020 },
            Section { name: ".text".into(), virtual_address: 0x3000, virtual_size: 0x100,
                raw_address: 0x1200, raw_size: 0x100, characteristics: 0x60000020 },
        ], opt_header_offset: 0, section_table_offset: 0, file_alignment: 0x200,
        section_alignment: 0x1000, size_of_headers: 0x200, loader_bound: vec![],
    };
    (pe, original)
}

#[test]
fn function_input_layout_keeps_snapshots_apart_from_guest_and_allocator_storage() {
    for locals in [0, 1, 0x100, 0x2100] {
        for spills in [0, 1, 17] {
            for calls in [false, true] {
                let f = FrameLayout::with_inputs(-locals, spills, calls,
                    vec![Reg::Rbx, Reg::R12], 3, vec![Reg::R9, Reg::Rcx, Reg::R9]);
                assert_eq!(f.inputs(), &[Reg::Rcx, Reg::R9]);
                let first = f.input_disp(Reg::Rcx).unwrap();
                let last = f.input_disp(Reg::R9).unwrap() + 8;
                assert!(first >= f.scratch_slot_disp(3) + 8);
                assert!(last <= f.frame_disp(-locals));
                assert_eq!(f.frame_size % 16, 8);
                for (lo, hi) in f.private_ranges() {
                    assert!(hi <= first || lo >= last);
                }
                assert!(f.input_disp(Reg::Rsp).is_none());
            }
        }
    }
}

#[test]
fn function_input_rsp_is_the_original_frame_address() {
    use crate::ir::expr::{Arena, BlockRef};
    let mut arena = Arena::new();
    let original_rsp = arena.init_reg(Reg::Rsp);
    let original_rcx = arena.init_reg(Reg::Rcx);
    let mut s = sched::Scheduler::new(&arena, BlockRef(0), DEFAULT_STACK_BASE);
    assert_eq!(s.value(original_rsp), Operand::Frame(0));
    assert!(matches!(s.value(original_rcx), Operand::Val(_)));
}

#[test]
fn function_input_loop_and_post_call_read_use_one_immutable_snapshot() {
    use crate::ir::sched::{CallTarget, ValId};
    let (pe, _) = fixture("monitor_r9");
    for kind in ["input_loop", "input_after_call"] {
        let mut original = CodeAssembler::new(64).unwrap();
        original.sub(rsp, 0x38).unwrap();
        let mut loop_label = original.create_label();
        original.mov(r10, r9).unwrap();
        original.mov(qword_ptr(rsp + 0x20), r9).unwrap();
        original.mov(r9d, 0x56).unwrap();
        if kind == "input_loop" {
            original.xor(eax, eax).unwrap();
            original.set_label(&mut loop_label).unwrap();
            original.inc(eax).unwrap();
            original.mov(qword_ptr(r10), rax).unwrap();
            original.cmp(eax, 3).unwrap();
            original.jne(loop_label).unwrap();
        } else {
            original.call(HELPER).unwrap();
            original.mov(r10, qword_ptr(rsp + 0x20)).unwrap();
            original.mov(qword_ptr(r10), 7).unwrap();
            original.mov(eax, 1).unwrap();
        }
        original.add(rsp, 0x38).unwrap();
        original.ret().unwrap();
        let original = original.assemble(START).unwrap();
        let id = |offset| BlockId { handler: START + offset, vip: None };
        let empty = |offset, terminator| SchedBlock {
            id: id(offset), instrs: vec![], exits: vec![], callee_set: vec![],
            indirect_tail: None, control: None, fused_cmp: None, terminator,
        };
        let cfg = Explorer::new(&pe, DEFAULT_STACK_BASE).recover(START);
        let return_term = cfg.blocks.iter().find_map(|b| {
            matches!(b.terminator, Terminator::Return { .. }).then(|| b.terminator.clone())
        }).unwrap();
        let (blocks, cfg) = if kind == "input_loop" {
            let mut entry = empty(0, Terminator::Jump(id(0x20)));
            entry.exits = vec![(Reg::Rax, Operand::Imm(0)), (Reg::R9, Operand::Imm(0x56))];
            let mut loop_body = empty(0x20, Terminator::Adopted { taken:id(0x40), not_taken:id(0x20) });
            loop_body.instrs = vec![
                Instr::FunctionInput { dst:ValId(0), reg:Reg::R9 },
                Instr::Bin { dst:ValId(1), op:crate::ir::expr::BinOp::Add,
                    a:Operand::Param(Reg::Rax), b:Operand::Imm(1), width:Width::W32, operand_width:Width::W32 },
                Instr::Store { addr:Operand::Val(ValId(0)), value:Operand::Val(ValId(1)), disp:0, width:Width::W64 },
            ];
            loop_body.exits = vec![(Reg::Rax, Operand::Val(ValId(1)))];
            loop_body.control = Some(Operand::Val(ValId(1)));
            loop_body.fused_cmp = Some(sched::FusedCmp { a:Operand::Val(ValId(1)), b:Operand::Imm(3),
                op:crate::ir::expr::BinOp::Eq, width:Width::W32 });
            let mut exit = empty(0x40, return_term);
            exit.exits = vec![(Reg::Rax, Operand::Param(Reg::Rax))];
            (vec![entry, loop_body, exit], cfg)
        } else {
            let mut entry = empty(0, Terminator::Jump(id(0x20)));
            entry.instrs = vec![Instr::Call { target:CallTarget::Direct(HELPER), args:vec![] }];
            let mut exit = empty(0x20, return_term);
            exit.instrs = vec![
                Instr::FunctionInput { dst:ValId(0), reg:Reg::R9 },
                Instr::Store { addr:Operand::Val(ValId(0)), value:Operand::Imm(7), disp:0, width:Width::W64 },
            ];
            exit.exits = vec![(Reg::Rax, Operand::Imm(1))];
            (vec![entry, exit], cfg)
        };
        let allocs:Vec<_> = blocks.iter().map(regalloc::allocate).collect();
        let emitted = emit_function_full(&cfg, &blocks, &allocs, OUTPUT, &pe, &CallRedirects::none()).unwrap();
        let layout = plan_frame(&blocks, &allocs);
        assert_eq!(layout.inputs(), &[Reg::R9]);
        assert_eq!(emitted.prologue.iter().filter(|p| matches!(p.step, PrologueStep::Alloc {..})).count(), 1);
        assert_eq!(emitted.prologue.iter().filter(|p| matches!(p.step, PrologueStep::Save {..})).count(), layout.saves().len());
        if let Some(dir) = std::env::var_os("TVM_FUNCTION_INPUT_FIXTURE_DIR") {
            let dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(format!("{kind}.original.bin")), original).unwrap();
            std::fs::write(dir.join(format!("{kind}.generated.bin")), &emitted.bytes).unwrap();
            std::fs::write(dir.join(format!("{kind}.xdata")), unwind_info(&emitted.prologue)).unwrap();
        }
    }
}

#[test]
fn function_input_forced_spill_preserves_borrowed_scratch_and_native_tail_arguments() {
    use crate::ir::regalloc::{Item, Segment};
    use crate::ir::sched::ValId;
    let (pe, _) = fixture("monitor_r9");
    let cfg = Explorer::new(&pe, DEFAULT_STACK_BASE).recover(START);
    let return_term = cfg.blocks.iter().find_map(|b| {
        matches!(b.terminator, Terminator::Return { .. }).then(|| b.terminator.clone())
    }).unwrap();
    for kind in ["input_spilled", "input_tail"] {
        let mut original = CodeAssembler::new(64).unwrap();
        if kind == "input_spilled" {
            original.mov(qword_ptr(r9), 7).unwrap();
            original.mov(eax, 1).unwrap();
            original.ret().unwrap();
        } else {
            original.mov(rcx, r9).unwrap();
            original.jmp(HELPER).unwrap();
        }
        let original = original.assemble(START).unwrap();
        let mut block = SchedBlock {
            id:cfg.entry, instrs:vec![Instr::FunctionInput { dst:ValId(0), reg:Reg::R9 }],
            exits:vec![], callee_set:vec![], indirect_tail:None, control:None, fused_cmp:None,
            terminator:return_term.clone(),
        };
        let alloc = if kind == "input_spilled" {
            block.instrs.push(Instr::Store { addr:Operand::Val(ValId(0)), value:Operand::Imm(7), disp:0, width:Width::W64 });
            block.exits.push((Reg::Rax, Operand::Imm(1)));
            let mut alloc = Alloc::default();
            alloc.slots = 1;
            alloc.segments.push(Segment { item:Item::Val(ValId(0)), start:0, end:4, loc:Loc::Spill(0) });
            // Deliberately consume every volatile register so with_spilled_dst
            // must borrow and preserve RAX instead of choosing a free scratch.
            for r in regalloc::VOLATILE {
                alloc.segments.push(Segment { item:Item::Entry(r), start:0, end:4, loc:Loc::Reg(r) });
            }
            assert!(alloc.free_volatile_at(0, &[]).is_none());
            alloc
        } else {
            block.exits.push((Reg::Rcx, Operand::Val(ValId(0))));
            block.terminator = Terminator::TailCall { target:HELPER };
            regalloc::allocate(&block)
        };
        let emitted = emit_function_full(&cfg, &[block], &[alloc], OUTPUT, &pe, &CallRedirects::none()).unwrap();
        if let Some(dir) = std::env::var_os("TVM_FUNCTION_INPUT_FIXTURE_DIR") {
            let dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(format!("{kind}.original.bin")), original).unwrap();
            std::fs::write(dir.join(format!("{kind}.generated.bin")), &emitted.bytes).unwrap();
            std::fs::write(dir.join(format!("{kind}.xdata")), unwind_info(&emitted.prologue)).unwrap();
        }
    }
}

#[test]
fn function_input_real_x86_fixtures_export_original_and_generated_bytes() {
    for kind in ["lock_c0", "lock_20", "monitor_r9"] {
        let (pe, original) = fixture(kind);
        let cfg = Explorer::new(&pe, DEFAULT_STACK_BASE).recover(START);
        assert_eq!(cfg.unresolved, 0);
        assert!(cfg.blocks.len() >= 3, "a distinct successor is necessary");
        let blocks = sched::schedule(&cfg);
        let allocs: Vec<_> = blocks.iter().map(regalloc::allocate).collect();
        let emitted = emit_function_full(&cfg, &blocks, &allocs, OUTPUT, &pe, &CallRedirects::none()).unwrap();
        assert!(!emitted.bytes.is_empty());
        if let Some(dir) = std::env::var_os("TVM_FUNCTION_INPUT_FIXTURE_DIR") {
            let dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(format!("{kind}.original.bin")), original).unwrap();
            std::fs::write(dir.join(format!("{kind}.generated.bin")), &emitted.bytes).unwrap();
            std::fs::write(dir.join(format!("{kind}.xdata")), unwind_info(&emitted.prologue)).unwrap();
            let lower = blocks.iter().map(|b| sched::render_block_with(b, &HashMap::new())).collect::<Vec<_>>().join("\n");
            std::fs::write(dir.join(format!("{kind}.lower.txt")), lower).unwrap();
        }
    }
}
