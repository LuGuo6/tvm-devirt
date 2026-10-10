//! Real instruction operands must survive CALL lifting and strict emission.
use super::*;
use crate::binary::pe::Section;
use crate::codegen::{self, CallRedirects, FrameLayout, ImageRange};
use crate::ir::{regalloc, sched};
use crate::vm::explore::Explorer;
use iced_x86::code_asm::*;

const IMAGE: u64 = 0x180000000;
const ENTRY: u64 = IMAGE + 0x1000;
const HELPER: u64 = IMAGE + 0x1800;
const STACK: u64 = 0x7fff00003000;
const GENERATED: u64 = IMAGE + 0x3000;

fn fixture(emit: impl FnOnce(&mut CodeAssembler)) -> PeFile {
    let mut a = CodeAssembler::new(64).unwrap();
    emit(&mut a);
    let bytes = a.assemble(ENTRY).unwrap();
    let mut data = vec![0xcc; 0x1200];
    data[0x200..0x200 + bytes.len()].copy_from_slice(&bytes);
    data[0xa00] = 0xc3;
    PeFile { data, image_base: IMAGE, entry_point_rva: 0x1000, size_of_image: 0x2000,
        sections: vec![Section { name: ".text".into(), virtual_address: 0x1000,
            virtual_size: 0x1000, raw_address: 0x200, raw_size: 0x1000,
            characteristics: 0x60000020 }], opt_header_offset: 0, section_table_offset: 0,
        file_alignment: 0x200, section_alignment: 0x1000, size_of_headers: 0x200,
        loader_bound: vec![] }
}

fn export(kind: &str, pe: &PeFile, cfg: &crate::ir::Cfg) -> Vec<u8> {
    assert_eq!(crate::binary::writeback::why_incomplete(cfg), None);
    let blocks = sched::schedule(cfg);
    let allocs: Vec<_> = blocks.iter().map(regalloc::allocate).collect();
    let bytes = codegen::emit_function(cfg, &blocks, &allocs, GENERATED, pe).unwrap();
    if let Some(out) = std::env::var_os("TVM_INDIRECT_CALL_FIXTURE_DIR") {
        let out = std::path::PathBuf::from(out);
        std::fs::create_dir_all(&out).unwrap();
        std::fs::write(out.join(format!("{kind}.original.bin")), &pe.data[0x200..]).unwrap();
        std::fs::write(out.join(format!("{kind}.generated.bin")), &bytes).unwrap();
    }
    bytes
}

#[test]
fn indirect_call_retains_register_input_before_call_clobber() {
    let pe = fixture(|a| {
        a.sub(rsp, 0x28).unwrap(); a.call(rdx).unwrap();
        a.add(rsp, 0x28).unwrap(); a.ret().unwrap();
    });
    let cfg = Explorer::new(&pe, STACK).recover(ENTRY);
    let target = cfg.blocks.iter().flat_map(|b| &b.events).find_map(|e| match e {
        Event::Call { target, .. } => *target, _ => None,
    }).expect("register CALL has a dynamic target");
    assert!(matches!(cfg.arena.op(target), Op::InitReg(Reg::Rdx)));
    export("register", &pe, &cfg);
}

#[test]
fn indirect_call_retains_memory_vtable_target_and_arguments() {
    let pe = fixture(|a| {
        a.sub(rsp, 0x28).unwrap(); a.mov(rax, qword_ptr(rcx)).unwrap();
        a.call(qword_ptr(rax + 0x20)).unwrap(); a.add(rsp, 0x28).unwrap(); a.ret().unwrap();
    });
    let cfg = Explorer::new(&pe, STACK).recover(ENTRY);
    let target = cfg.blocks.iter().flat_map(|b| &b.events).find_map(|e| match e {
        Event::Call { target, .. } => *target, _ => None,
    }).expect("memory CALL has a dynamic target");
    assert!(matches!(cfg.arena.op(target), Op::Load(..)));
    assert!(cfg.blocks.iter().flat_map(|b| &b.events).any(|e| match e {
        Event::Call { args, .. } => args.iter().any(|(r, v)|
            *r == Reg::Rcx && matches!(cfg.arena.op(*v), Op::InitReg(Reg::Rcx))),
        _ => false,
    }), "an untouched incoming this pointer is still a callee argument");
    let bytes = export("vtable", &pe, &cfg);
    assert!(iced_x86::Decoder::with_ip(64, &bytes, GENERATED, 0).into_iter().any(|i|
        i.mnemonic() == Mnemonic::Call
            && matches!(i.op0_kind(), OpKind::Register | OpKind::Memory)));
}

#[test]
fn indirect_call_keeps_target_while_replacing_its_initial_argument_register() {
    let pe = fixture(|a| {
        a.sub(rsp, 0x28).unwrap(); a.mov(r10, rcx).unwrap(); a.mov(rcx, rdx).unwrap();
        a.call(r10).unwrap(); a.add(rsp, 0x28).unwrap(); a.ret().unwrap();
    });
    let cfg = Explorer::new(&pe, STACK).recover(ENTRY);
    export("target_argument_overlap", &pe, &cfg);
}

#[test]
fn indirect_call_preserves_vtable_expression_from_a_prior_helper_result() {
    let pe = fixture(|a| {
        a.sub(rsp, 0x28).unwrap(); a.call(HELPER).unwrap();
        a.mov(rdx, qword_ptr(rax)).unwrap(); a.mov(rdx, qword_ptr(rdx + 0x20)).unwrap();
        a.mov(rcx, rax).unwrap(); a.call(rdx).unwrap();
        a.add(rsp, 0x28).unwrap(); a.ret().unwrap();
    });
    let cfg = Explorer::new(&pe, STACK).recover(ENTRY);
    let targets: Vec<_> = cfg.blocks.iter().flat_map(|b| &b.events).filter_map(|e| match e {
        Event::Call { target, .. } => Some(target), _ => None,
    }).collect();
    assert_eq!(targets.len(), 2);
    assert!(targets[1].is_some_and(|r| matches!(cfg.arena.op(r), Op::Load(..))));
    export("helper_vtable", &pe, &cfg);
}

#[test]
fn indirect_call_missing_target_rejects_instead_of_inventing_a_trap() {
    let pe = fixture(|a| { a.xor(eax, eax).unwrap(); a.call(rax).unwrap(); a.ret().unwrap(); });
    let cfg = Explorer::new(&pe, STACK).recover(ENTRY);
    assert!(crate::binary::writeback::why_incomplete(&cfg).unwrap().contains("unknown call target"));
    let mut asm = CodeAssembler::new(64).unwrap();
    let result = codegen::emit_call(&mut asm, &regalloc::Alloc::default(),
        &sched::CallTarget::Unknown { site: ENTRY }, 0, ImageRange::none(),
        &FrameLayout::new(0, 0, true), &CallRedirects::none());
    assert!(result.is_err());
    assert!(asm.instructions().is_empty());
}

#[test]
fn indirect_call_missing_allocation_rejects_instead_of_inventing_a_trap() {
    let mut asm = CodeAssembler::new(64).unwrap();
    let result = codegen::emit_call(&mut asm, &regalloc::Alloc::default(),
        &sched::CallTarget::Indirect(sched::Operand::Val(sched::ValId(99))), 0,
        ImageRange::none(), &FrameLayout::new(0, 0, true), &CallRedirects::none());
    assert!(result.is_err());
    assert!(asm.instructions().is_empty());
}

#[test]
fn indirect_call_known_noreturn_call_keeps_its_real_target() {
    let pe = fixture(|a| { a.call(HELPER).unwrap(); a.int3().unwrap(); });
    let cfg = Explorer::new(&pe, STACK).recover(ENTRY);
    assert_eq!(crate::binary::writeback::why_incomplete(&cfg), None);
    assert!(cfg.blocks.iter().flat_map(|b| &b.events).any(|e| matches!(e,
        Event::Call { target: Some(target), .. } if cfg.arena.as_const(*target) == Some(HELPER))));
}

#[test]
fn indirect_call_does_not_resurrect_abi_inputs_after_a_prior_call() {
    let mut pe = fixture(|a| {
        a.sub(rsp, 0x28).unwrap(); a.call(HELPER).unwrap(); a.call(rax).unwrap();
        a.add(rsp, 0x28).unwrap(); a.ret().unwrap();
    });
    // This test concerns ABI clobbers. A literal RET helper also has a real
    // incoming-RAX dependency, so use a helper that defines its return value.
    pe.data[0xa00..0xa03].copy_from_slice(&[0x31, 0xc0, 0xc3]);
    let cfg = Explorer::new(&pe, STACK).recover(ENTRY);
    let arguments: Vec<_> = cfg.blocks.iter().flat_map(|b| &b.events).filter_map(|e| match e {
        Event::Call { args, .. } => Some(args), _ => None,
    }).collect();
    assert_eq!(arguments.len(), 2);
    assert_eq!(arguments[0].len(), 4, "the first helper sees unchanged ABI inputs");
    assert!(arguments[1].is_empty(), "post-helper clobbers are not original inputs");
}

#[test]
fn native_rax_input_preserves_real_helper_reads_and_passthrough_returns() {
    for (kind, helper, incoming) in [
        ("rax_ret", vec![0xc3], true),
        ("rax_read", vec![0x48, 0x89, 0x02, 0x31, 0xc0, 0xc3], true),
        ("rax_al", vec![0xb0, 0x42, 0xc3], true),
        ("rax_ax_zero", vec![0x66, 0x31, 0xc0, 0xc3], true),
        ("rax_branch", vec![0x85, 0xc9, 0x75, 0x01, 0xc3, 0x31, 0xc0, 0xc3], true),
        ("rax_check", vec![0x85, 0xc9, 0x75, 0x01, 0xc3, 0x0f, 0x0b], true),
        ("rax_eax_zero", vec![0x31, 0xc0, 0xc3], false),
    ] {
        let mut pe = fixture(|a| {
            a.sub(rsp, 0x28).unwrap();
            a.mov(rax, r8).unwrap();
            a.add(rax, 123).unwrap();
            a.mov(rcx, r9).unwrap();
            a.call(HELPER).unwrap();
            a.add(rsp, 0x28).unwrap();
            a.ret().unwrap();
        });
        pe.data[0xa00..0xa00 + helper.len()].copy_from_slice(&helper);
        let cfg = Explorer::new(&pe, STACK).recover(ENTRY);
        let args = cfg.blocks.iter().flat_map(|b| &b.events).find_map(|e| match e {
            Event::Call { args, .. } => Some(args), _ => None,
        }).unwrap();
        assert_eq!(args.iter().any(|(r, _)| *r == Reg::Rax), incoming, "{kind}");
        export(kind, &pe, &cfg);
    }
}
