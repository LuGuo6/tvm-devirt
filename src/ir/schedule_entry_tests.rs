//! A native continuation below the VM entry must not become the scheduling root.
use super::*;
use crate::binary::pe::PeFile;
use crate::ir::expr::BlockRef;
use crate::ir::regalloc;

fn id(handler: u64) -> BlockId {
    BlockId { handler, vip: None }
}

fn fixture() -> Cfg {
    let mut arena = Arena::new();
    let input = arena.init_reg(Reg::Rcx);
    let destination = arena.opaque("return_address", Width::W64);
    let root_callee = arena.constant(0x8000, Width::W64);
    let error_callee = arena.constant(0x9000, Width::W64);
    let one = arena.constant(1, Width::W64);
    let two = arena.constant(2, Width::W64);
    let call = |target| Event::Call {
        target: Some(target),
        site: 0x3000,
        rsp: Some(0x7fff00000fd8),
        import_slot: None,
        args: vec![],
        ret: None,
    };
    Cfg {
        entry: id(0x3000),
        arena,
        total_steps: 3,
        unresolved: 0,
        back_edges: vec![],
        trivial_phis_removed: 0,
        timed_out: false,
        stack_base: 0x7fff00001000,
        native_entry_proofs: Default::default(), vm_range: (0, 0),
        // Exactly the order produced by recover_pass_with: handler address.
        blocks: vec![
            Block {
                id: id(0x1000),
                block_ref: BlockRef(1),
                params: vec![],
                events: vec![],
                from_vm_context: false,
                exit_regs: vec![(Reg::Rax, one)],
                terminator: Terminator::Return { dest: destination },
                cost: 1,
                preds: vec![id(0x3000)],
            },
            Block {
                id: id(0x3000),
                block_ref: BlockRef(0),
                params: vec![],
                events: vec![call(root_callee)],
                from_vm_context: false,
                exit_regs: vec![],
                terminator: Terminator::Branch {
                    predicate: input,
                    taken: id(0x4000),
                    not_taken: id(0x1000),
                },
                cost: 1,
                preds: vec![],
            },
            Block {
                id: id(0x4000),
                block_ref: BlockRef(2),
                params: vec![],
                events: vec![call(error_callee)],
                from_vm_context: false,
                exit_regs: vec![(Reg::Rax, two)],
                terminator: Terminator::Return { dest: destination },
                cost: 1,
                preds: vec![id(0x3000)],
            },
        ],
    }
}

fn pe() -> PeFile {
    PeFile {
        data: vec![],
        image_base: 0,
        entry_point_rva: 0,
        size_of_image: 0,
        sections: vec![],
        opt_header_offset: 0,
        section_table_offset: 0,
        file_alignment: 0x200,
        section_alignment: 0x1000,
        size_of_headers: 0,
        loader_bound: vec![],
    }
}

#[test]
fn schedule_entry_retains_real_root_branches_and_native_calls_in_emitted_bytes() {
    let cfg = fixture();
    let scheduled = schedule(&cfg);
    assert_eq!(scheduled[0].id, cfg.entry);
    assert_eq!(
        scheduled.len(),
        3,
        "lower-address return must not discard the true root and sibling"
    );
    let allocations: Vec<_> = scheduled.iter().map(regalloc::allocate).collect();
    let bytes =
        crate::codegen::emit_function(&cfg, &scheduled, &allocations, 0x100000, &pe()).unwrap();
    let instructions: Vec<_> = iced_x86::Decoder::with_ip(64, &bytes, 0x100000, 0)
        .into_iter()
        .collect();
    assert!(instructions.iter().all(|i| !i.is_invalid()));
    let calls: Vec<_> = instructions
        .iter()
        .filter(|i| i.mnemonic() == iced_x86::Mnemonic::Call)
        .map(|i| i.near_branch64())
        .collect();
    assert_eq!(
        calls,
        [0x8000, 0x9000],
        "the entry's call must precede the retained error-path call"
    );
    assert!(
        instructions
            .iter()
            .any(|i| i.flow_control() == iced_x86::FlowControl::ConditionalBranch)
    );
}

#[test]
fn schedule_entry_keeps_a_high_address_empty_entry_jump() {
    let mut cfg = fixture();
    let entry = cfg.blocks.iter_mut().find(|b| b.id == cfg.entry).unwrap();
    entry.events.clear();
    entry.terminator = Terminator::Jump(id(0x1000));
    cfg.blocks.retain(|block| block.id != id(0x4000));
    let scheduled = schedule(&cfg);
    assert_eq!(scheduled[0].id, cfg.entry);
    assert!(scheduled.iter().any(|b| b.id == id(0x1000)));
}

#[test]
fn schedule_entry_codegen_rejects_a_misordered_or_missing_root() {
    let cfg = fixture();
    let mut scheduled = schedule(&cfg);
    scheduled.swap(0, 1);
    for omit_entry in [false, true] {
        let mut broken = scheduled.clone();
        if omit_entry {
            broken.retain(|b| b.id != cfg.entry);
        }
        let allocations: Vec<_> = broken.iter().map(regalloc::allocate).collect();
        let error = crate::codegen::emit_function(&cfg, &broken, &allocations, 0x100000, &pe())
            .unwrap_err();
        assert!(error.to_string().contains("does not match recovered entry"));
    }
}
