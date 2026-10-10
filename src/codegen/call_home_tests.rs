//! Backend-only qword home fixtures. Origin/dominance acceptance is tested by
//! the recovery/scheduler layer; these exercise actual lowering and frame use.
use super::*;
use crate::ir::expr::{Arena, BinOp, BlockRef};
use crate::ir::regalloc::Segment;
use crate::ir::sched::ValId;
use crate::ir::Block;

const IMAGE: u64 = 0x180000000;
const BODY: u64 = IMAGE + 0x5000;
const HELPER_A: u64 = IMAGE + 0x1800;
const HELPER_B: u64 = IMAGE + 0x1810;

fn id(n: u64) -> BlockId { BlockId { handler: IMAGE + 0x1000 + 0x100*n, vip: None } }
fn val(n: u32) -> Operand { Operand::Val(ValId(n)) }

fn scheduled(n: u64, instrs: Vec<Instr>, terminator: Terminator) -> SchedBlock {
    SchedBlock { id: id(n), instrs, exits: vec![], callee_set: vec![],
        indirect_tail: None, control: None, terminator, fused_cmp: None }
}

fn bin(dst: u32, op: BinOp, a: Operand, b: Operand) -> Instr {
    Instr::Bin { dst: ValId(dst), op, a, b, width: Width::W64, operand_width: Width::W64 }
}

fn fixture(looping: bool) -> (Cfg, Vec<SchedBlock>, PeFile) {
    let mut arena = Arena::new();
    let ret = arena.opaque("return_address", Width::W64);
    let predicate = arena.constant(1, Width::W8);
    let mut first = vec![
        Instr::Store { addr: Operand::OutArg(32), value: Operand::Imm(0x23), disp: 0, width: Width::W64 },
        Instr::Call { target: CallTarget::Direct(HELPER_A), args: vec![] },
        Instr::Opaque { dst: ValId(0), tag: "call_ret", width: Width::W64, at: Some(Reg::Rax) },
        Instr::WriteCallHome { slot: 0, value: val(0) },
        Instr::Store { addr: Operand::Frame(-0x30), value: Operand::Imm(0x11223344), disp: 0, width: Width::W64 },
    ];
    let mut blocks;
    if looping {
        first.push(Instr::WriteCallHome { slot: 1, value: Operand::Imm(0) });
        let mut repeated = scheduled(1, vec![
            Instr::ReadCallHome { dst: ValId(0), slot: 1 },
            Instr::Call { target: CallTarget::Direct(HELPER_B), args: vec![] },
            bin(1, BinOp::Add, val(0), Operand::Imm(1)),
            Instr::WriteCallHome { slot: 1, value: val(1) },
            Instr::Bin { dst: ValId(2), op: BinOp::Ult, a: val(1), b: Operand::Imm(3),
                width: Width::W8, operand_width: Width::W64 },
        ], Terminator::Branch { predicate, taken: id(1), not_taken: id(2) });
        repeated.control = Some(val(2));
        blocks = vec![scheduled(0, first, Terminator::Jump(id(1))), repeated];
    } else {
        first.extend([
            Instr::Call { target: CallTarget::Direct(HELPER_B), args: vec![] },
            Instr::Opaque { dst: ValId(1), tag: "call_ret", width: Width::W64, at: Some(Reg::Rax) },
            Instr::WriteCallHome { slot: 1, value: val(1) },
        ]);
        blocks = vec![scheduled(0, first, Terminator::Jump(id(1)))];
    }
    let last = blocks.len() as u64;
    let mut tail = scheduled(last, vec![
        Instr::ReadCallHome { dst: ValId(0), slot: 0 },
        Instr::ReadCallHome { dst: ValId(1), slot: 1 },
        bin(2, BinOp::Xor, val(0), val(1)),
        Instr::FunctionInput { dst: ValId(3), reg: Reg::Rdx },
        bin(4, BinOp::Xor, val(2), val(3)),
        Instr::Load { dst: ValId(5), addr: Operand::Frame(-0x30), disp: 0, width: Width::W64 },
        bin(6, BinOp::Xor, val(4), val(5)),
    ], Terminator::Return { dest: ret });
    tail.exits.push((Reg::Rax, val(6)));
    blocks.push(tail);
    let cfg_blocks = blocks.iter().enumerate().map(|(n,b)| Block {
        id: b.id, block_ref: BlockRef(n as u32), params: vec![], events: vec![],
        from_vm_context: false, exit_regs: vec![], terminator: b.terminator.clone(),
        cost: 1, preds: if n == 0 { vec![] } else { vec![id(n as u64-1)] },
    }).collect();
    let cfg = Cfg { entry: id(0), arena, blocks: cfg_blocks, total_steps: 1,
        unresolved: 0, back_edges: if looping {vec![(id(1),id(1))]} else {vec![]},
        trivial_phis_removed: 0, timed_out: false, stack_base: 0x7fff00003000, native_entry_proofs: Default::default(), vm_range: (0,0) };
    let pe = PeFile { data: vec![], image_base: IMAGE, entry_point_rva: 0x1000,
        size_of_image: 0x10000, sections: vec![], opt_header_offset: 0,
        section_table_offset: 0, file_alignment: 0x200, section_alignment: 0x1000,
        size_of_headers: 0, loader_bound: vec![] };
    (cfg, blocks, pe)
}

#[test]
fn call_home_frame_is_disjoint_from_all_other_regions() {
    for slots in [1, 2, 17, 600] {
        let layout = FrameLayout::with_call_homes(-0x230, 5, true,
            vec![Reg::Rbx, Reg::R12], 3, vec![Reg::Rcx, Reg::Rdx], slots);
        let mut occupied = HashSet::new();
        for n in 0..7 { assert!(occupied.insert(n*8)); }
        for n in 0..5 { assert!(occupied.insert(layout.spill_disp(n))); }
        for &r in layout.saves() { assert!(occupied.insert(layout.save_disp(r).unwrap())); }
        for n in 0..SCRATCH_SLOTS as u32 { assert!(occupied.insert(layout.scratch_slot_disp(n))); }
        for &r in layout.inputs() { assert!(occupied.insert(layout.input_disp(r).unwrap())); }
        for n in 0..slots {
            let d = layout.call_home_disp(n).unwrap();
            assert!(occupied.insert(d));
            assert!(d+8 <= layout.frame_disp(-0x230));
            assert!(layout.private_ranges().iter().all(|&(lo,hi)| d+8 <= lo || d >= hi));
        }
        assert!(layout.call_home_disp(slots).is_none());
        assert_eq!(layout.frame_size % 16, 8);
    }
}

#[test]
fn call_home_lowering_rejects_a_slot_missing_from_the_frame() {
    let instrs = [Instr::ReadCallHome { dst: ValId(0), slot: 1 },
        Instr::WriteCallHome { slot: 1, value: Operand::Imm(123) }];
    let layout = FrameLayout::with_call_homes(0,0,false,vec![],0,vec![],1);
    for instr in instrs {
        let mut asm = CodeAssembler::new(64).unwrap();
        assert!(emit_instr(&mut asm, &Alloc::default(), &instr, 0, &layout,
            ImageRange::empty(), None, &CallRedirects::none()).is_err());
    }
}

#[test]
fn call_home_spilled_predicate_reads_only_its_defined_width() {
    for width in [Width::W8, Width::W16, Width::W32, Width::W64] {
        let mut arena = Arena::new();
        let predicate = arena.constant(1,Width::W8);
        let mut block = scheduled(0,vec![Instr::Opaque { dst: ValId(0), tag: "fixture",
            width, at: Some(Reg::Rcx) }],
            Terminator::Branch { predicate, taken: id(1), not_taken: id(2) });
        block.control=Some(val(0));
        let layout=FrameLayout::new(0,1,false);
        let alloc=Alloc { segments: vec![Segment { item: Item::Val(ValId(0)), start: 0,
            end: 3, loc: Loc::Spill(0) }], slots:1, ..Alloc::default() };
        let mut asm=CodeAssembler::new(64).unwrap();
        asm.sub(rsp,layout.frame_size as i32).unwrap();
        asm.mov(qword_ptr(rsp+layout.spill_disp(0) as i32),rcx).unwrap();
        let mut yes=asm.create_label(); let mut no=asm.create_label();
        emit_branch_jcc(&mut asm,&alloc,&block,Some(yes),Some(no),&layout).unwrap();
        asm.set_label(&mut yes).unwrap(); asm.mov(eax,1).unwrap();
        asm.add(rsp,layout.frame_size as i32).unwrap(); asm.ret().unwrap();
        asm.set_label(&mut no).unwrap(); asm.xor(eax,eax).unwrap();
        asm.add(rsp,layout.frame_size as i32).unwrap(); asm.ret().unwrap();
        let bytes=asm.assemble(BODY).unwrap();
        let comparison=iced_x86::Decoder::with_ip(64,&bytes,BODY,0).into_iter()
            .find(|i| i.mnemonic()==iced_x86::Mnemonic::Cmp).unwrap();
        assert_eq!(comparison.memory_size().size(),width.bytes() as usize);
        if let Some(dir)=std::env::var_os("TVM_CALL_HOME_FIXTURE_DIR") {
            let dir=std::path::PathBuf::from(dir).join("predicates");
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(format!("w{}.bin",width.bits())),bytes).unwrap();
        }
    }
}

#[test]
fn call_home_cross_block_calls_and_loops_export_actual_bytes_and_unwind() {
    for (name, looping, spilled) in [("registers",false,false), ("spills",false,true),
        ("loop-registers",true,false), ("loop-spills",true,true)] {
        let (cfg, blocks, pe) = fixture(looping);
        let allocs: Vec<_> = blocks.iter().map(|block| {
            let mut alloc = regalloc::allocate(block);
            if spilled {
                // No block-entry parameters in this fixture. Giving every value
                // a distinct spill removes incidental register assignment from
                // the call-home correctness claim.
                let items: HashSet<_> = alloc.segments.iter().map(|s| s.item).collect();
                let mut items: Vec<_> = items.into_iter().collect();
                items.sort_by_key(|item| format!("{item:?}"));
                alloc.segments = items.iter().enumerate().map(|(n,&item)| Segment {
                    item, start: 0, end: block.instrs.len()+2, loc: Loc::Spill(n as u32),
                }).collect();
                alloc.copies.clear();
                alloc.slots = items.len() as u32;
            }
            alloc
        }).collect();
        let layout = plan_frame(&blocks,&allocs);
        assert!(layout.call_home_disp(1).is_some());
        assert!(layout.call_home_disp(2).is_none());
        assert!(layout.input_disp(Reg::Rdx).is_some());
        let emitted = emit_function_full(&cfg,&blocks,&allocs,BODY,&pe,&CallRedirects::none()).unwrap();
        let prologue = measure_prologue(&emitted.bytes,&layout).unwrap();
        assert!(matches!(prologue[0].step, PrologueStep::Alloc { bytes } if bytes == layout.frame_size));
        assert_eq!(prologue.len(),layout.saves().len()+1);
        let instructions: Vec<_> = iced_x86::Decoder::with_ip(64,&emitted.bytes,BODY,0).into_iter().collect();
        for slot in 0..2 {
            let d = layout.call_home_disp(slot).unwrap() as u64;
            assert!(instructions.iter().any(|i| i.mnemonic()==iced_x86::Mnemonic::Mov
                && i.op0_kind()==iced_x86::OpKind::Memory && i.memory_base()==iced_x86::Register::RSP
                && i.memory_displacement64()==d), "{name}: cross-block home store was removed");
        }
        if let Some(dir) = std::env::var_os("TVM_CALL_HOME_FIXTURE_DIR") {
            let dir = std::path::PathBuf::from(dir); std::fs::create_dir_all(&dir).unwrap();
            let mut original = CodeAssembler::new(64).unwrap();
            original.push(rbx).unwrap(); original.push(rsi).unwrap(); original.push(rdi).unwrap();
            original.sub(rsp,0x40).unwrap(); original.mov(rsi,rdx).unwrap();
            original.mov(qword_ptr(rsp+0x20),0x23).unwrap();
            original.call(HELPER_A).unwrap(); original.mov(rbx,rax).unwrap();
            original.mov(qword_ptr(rsp+0x30),0x11223344).unwrap();
            if looping {
                original.xor(edi,edi).unwrap();
                let mut again = original.create_label(); original.set_label(&mut again).unwrap();
                original.call(HELPER_B).unwrap(); original.inc(edi).unwrap();
                original.cmp(edi,3).unwrap(); original.jne(again).unwrap();
                original.xor(rbx,rdi).unwrap();
            } else {
                original.call(HELPER_B).unwrap(); original.xor(rbx,rax).unwrap();
            }
            original.xor(rbx,rsi).unwrap(); original.xor(rbx,qword_ptr(rsp+0x30)).unwrap();
            original.mov(rax,rbx).unwrap(); original.add(rsp,0x40).unwrap();
            original.pop(rdi).unwrap(); original.pop(rsi).unwrap(); original.pop(rbx).unwrap();
            original.ret().unwrap();
            std::fs::write(dir.join(format!("{name}.original.bin")),original.assemble(IMAGE+0x1000).unwrap()).unwrap();
            std::fs::write(dir.join(format!("{name}.bin")),&emitted.bytes).unwrap();
            std::fs::write(dir.join(format!("{name}.xdata")),unwind_info(&prologue)).unwrap();
            std::fs::write(dir.join(format!("{name}.json")),format!(
                "{{\"kind\":\"{name}\",\"looping\":{looping},\"frame_size\":{},\"prologue_size\":{},\"code_size\":{},\"home0\":{},\"home1\":{}}}",
                layout.frame_size,prologue.last().unwrap().end,emitted.bytes.len(),
                layout.call_home_disp(0).unwrap(),layout.call_home_disp(1).unwrap())).unwrap();
        }
    }
}

#[test]
fn review_call_home_dominance_schedule_exports_actual_bytes() {
    use crate::ir::expr::CallOrigin;
    use crate::ir::lift::Event;
    for looping in [false,true] {
        let mut arena=Arena::new();
        let input=arena.init_reg(Reg::Rdx);
        let zero=arena.constant(0,Width::W64);
        let retaddr=arena.opaque("return_address",Width::W64);
        let origin=CallOrigin {block:id(0),event_index:0,site:IMAGE+0x1005};
        let old=arena.opaque("call_ret",Width::W64);arena.bind_call_origin(old,origin);
        let repeated_origin=CallOrigin {block:id(1),event_index:0,site:IMAGE+0x1105};
        let repeated=arena.opaque("call_ret",Width::W64);arena.bind_call_origin(repeated,repeated_origin);
        let target_a=arena.constant(HELPER_A,Width::W64);
        let target_b=arena.constant(HELPER_B,Width::W64);
        let target_c=arena.constant(HELPER_B+0x10,Width::W64);
        let predicate=arena.bin(BinOp::Eq,if looping {repeated} else {input},zero);
        let output=arena.bin(BinOp::Xor,old,input);
        let call=|target,site,ret,args| Event::Call {target:Some(target),site,rsp:None,
            import_slot:None,args,ret};
        let block=|n,events,terminator,exits| Block {id:id(n),block_ref:BlockRef(n as u32),
            params:vec![],events,from_vm_context:false,exit_regs:exits,terminator,cost:1,preds:vec![]};
        let a_args=[Reg::Rcx,Reg::Rdx,Reg::R8,Reg::R9].into_iter()
            .map(|r|(r,arena.init_reg(r))).collect();
        let first=block(0,vec![call(target_a,origin.site,Some(old),a_args)],if looping {
            Terminator::Jump(id(1))
        } else {Terminator::Branch {predicate,taken:id(1),not_taken:id(2)}},vec![]);
        let second=block(1,vec![call(target_b,repeated_origin.site,Some(repeated),vec![(Reg::Rcx,old)])],
            if looping {Terminator::Branch {predicate,taken:id(3),not_taken:id(1)}}
            else {Terminator::Jump(id(3))},vec![]);
        let mut blocks=vec![first,second];
        if !looping {blocks.push(block(2,vec![call(target_c,IMAGE+0x1205,None,vec![(Reg::Rcx,old)])],
            Terminator::Jump(id(3)),vec![]));}
        blocks.push(block(3,vec![],Terminator::Return {dest:retaddr},vec![(Reg::Rax,output)]));
        let cfg=Cfg {entry:id(0),arena,blocks,total_steps:1,unresolved:0,
            back_edges:if looping {vec![(id(1),id(1))]} else {vec![]},trivial_phis_removed:0,
            timed_out:false,stack_base:0x7fff00003000,native_entry_proofs:Default::default(),vm_range:(0,0)};
        let scheduled=crate::ir::sched::schedule(&cfg);
        assert_eq!(scheduled.iter().flat_map(|b|&b.instrs).filter(|i|matches!(i,Instr::WriteCallHome {..})).count(),1);
        assert!(!scheduled.iter().flat_map(|b|&b.instrs).any(|i|matches!(i,Instr::Opaque {tag:"call_ret",at:None,..})));
        let allocs=scheduled.iter().map(regalloc::allocate).collect::<Vec<_>>();
        let (_,_,pe)=fixture(false);
        let emitted=emit_function_full(&cfg,&scheduled,&allocs,BODY,&pe,&CallRedirects::none()).unwrap();
        if let Some(dir)=std::env::var_os("TVM_DOMINANCE_FIXTURE_DIR") {
            let dir=std::path::PathBuf::from(dir);std::fs::create_dir_all(&dir).unwrap();
            let name=if looping {"use-loop"} else {"diamond"};
            let mut asm=CodeAssembler::new(64).unwrap();
            asm.push(rbx).unwrap();asm.push(rsi).unwrap();asm.sub(rsp,0x28).unwrap();
            asm.mov(rsi,rdx).unwrap();asm.call(HELPER_A).unwrap();asm.mov(rbx,rax).unwrap();
            if looping {
                let mut again=asm.create_label();asm.set_label(&mut again).unwrap();
                asm.mov(rcx,rbx).unwrap();asm.call(HELPER_B).unwrap();asm.test(rax,rax).unwrap();asm.jnz(again).unwrap();
            } else {
                let mut yes=asm.create_label();let mut join=asm.create_label();
                asm.test(rsi,rsi).unwrap();asm.jz(yes).unwrap();
                asm.mov(rcx,rbx).unwrap();asm.call(HELPER_B+0x10).unwrap();asm.jmp(join).unwrap();
                asm.set_label(&mut yes).unwrap();asm.mov(rcx,rbx).unwrap();asm.call(HELPER_B).unwrap();asm.set_label(&mut join).unwrap();
            }
            asm.mov(rax,rbx).unwrap();asm.xor(rax,rsi).unwrap();asm.add(rsp,0x28).unwrap();asm.pop(rsi).unwrap();asm.pop(rbx).unwrap();asm.ret().unwrap();
            std::fs::write(dir.join(format!("{name}.original.bin")),asm.assemble(IMAGE+0x1000).unwrap()).unwrap();
            std::fs::write(dir.join(format!("{name}.bin")),emitted.bytes).unwrap();
        }
    }
}
