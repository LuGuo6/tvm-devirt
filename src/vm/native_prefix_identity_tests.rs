use super::*;
use crate::binary::pe::Section;
use iced_x86::code_asm::*;

const IMAGE: u64 = 0x180000000;
const ENTRY: u64 = IMAGE + 0x1000;
const ARM: u64 = IMAGE + 0x1100;
const EXIT: u64 = IMAGE + 0x1300;
const STACK: u64 = 0x7fff00003000;

fn id(handler: u64) -> BlockId { BlockId { handler, vip: None } }

fn put(pe: &mut PeFile, address: u64, emit: impl FnOnce(&mut CodeAssembler)) {
    let mut a = CodeAssembler::new(64).unwrap();
    emit(&mut a);
    let bytes = a.assemble(address).unwrap();
    let offset = (address - ENTRY + 0x200) as usize;
    pe.data[offset..offset + bytes.len()].copy_from_slice(&bytes);
}

fn image(unsupported_prefix: bool) -> PeFile {
    let mut pe = PeFile {
        data: vec![0xcc; 0x1200], image_base: IMAGE, entry_point_rva: 0x1000,
        size_of_image: 0x3000,
        sections: vec![Section { name: ".tvm0".into(), virtual_address: 0x1000,
            virtual_size: 0x1000, raw_address: 0x200, raw_size: 0x1000,
            characteristics: 0x60000020 }],
        opt_header_offset: 0, section_table_offset: 0, file_alignment: 0x200,
        section_alignment: 0x1000, size_of_headers: 0x200, loader_bound: vec![],
    };
    put(&mut pe, ENTRY, |a| {
        if unsupported_prefix { a.pushfq().unwrap(); a.popfq().unwrap(); }
        a.test(ecx, ecx).unwrap();
        a.je(ARM).unwrap();
        a.ret().unwrap();
    });
    put(&mut pe, ARM, |a| {
        for (i, r) in [rax, rbx, rcx, rdx, rsp, rbp, rsi, rdi,
            r8, r9, r10, r11, r12, r13, r14, r15].into_iter().enumerate() {
            a.mov(qword_ptr(rsp - 0x200 + i as i32 * 8), r).unwrap();
        }
        a.test(edx, edx).unwrap();
        a.je(EXIT).unwrap();
        a.ret().unwrap();
    });
    put(&mut pe, EXIT, |a| { a.ret().unwrap(); });
    pe
}

#[test]
fn native_prefix_identity_late_context_is_not_entry_context() {
    let pe = image(false);
    let cfg = Explorer::new(&pe, STACK).recover_pass_with(
        ENTRY, &HashSet::new(), None, None, None);
    let arm = cfg.block(id(ARM)).unwrap();
    assert_eq!(arm.preds, [id(ENTRY)]);
    assert!(arm.events.iter().any(|e| matches!(e,
        Event::Store { region: crate::ir::lift::Region::VmScratch, .. })),
        "the child must actually build a recognized VM image");
    assert!(arm.from_vm_context, "exit registers still come from the newly built VM image");
    assert!(cfg.native_entry_proofs.contains(&id(ARM)),
        "known native-history entry must not acquire its later VM-context identity");
}

#[test]
fn native_prefix_identity_unsupported_history_remains_unidentified() {
    let pe = image(true);
    let cfg = Explorer::new(&pe, STACK).recover_pass_with(
        ENTRY, &HashSet::new(), None, None, None);
    assert!(cfg.block(id(ARM)).unwrap().from_vm_context);
    assert!(validate(&cfg).unidentified_blocks.contains(&id(ARM)));
}

#[test]
fn native_prefix_identity_preknown_unused_vip_slot_is_not_active_vm_state() {
    let pe = image(false);
    let cfg = Explorer::new(&pe, STACK).recover_pass_with(
        ENTRY, &HashSet::new(), None, Some(0x100), None);
    assert!(cfg.native_entry_proofs.contains(&id(ARM)));
}

#[test]
fn native_prefix_identity_late_unproven_incoming_revokes_the_first_path() {
    const OTHER: u64 = IMAGE + 0x1400;
    for dirty_fallthrough in [true, false] {
        let mut pe = image(false);
        put(&mut pe, ENTRY, |a| {
            a.test(ecx, ecx).unwrap(); a.je(OTHER).unwrap();
            if dirty_fallthrough { a.pushfq().unwrap(); a.popfq().unwrap(); }
            a.test(esi, esi).unwrap(); a.je(ARM).unwrap(); a.ret().unwrap();
        });
        put(&mut pe, OTHER, |a| {
            if !dirty_fallthrough { a.pushfq().unwrap(); a.popfq().unwrap(); }
            a.test(esi, esi).unwrap(); a.je(ARM).unwrap(); a.ret().unwrap();
        });
        let cfg = Explorer::new(&pe, STACK).recover_pass_with(
            ENTRY, &HashSet::new(), None, None, None);
        let arm = cfg.block(id(ARM)).unwrap();
        assert_eq!(arm.preds.len(), 2);
        assert!(matches!(arm.terminator, Terminator::Branch { .. }),
            "independent EDX must remain symbolic after the ESI incoming branch");
        assert!(!cfg.native_entry_proofs.contains(&id(ARM)), "one incoming has unsupported history; dirty_fallthrough={dirty_fallthrough}");
        assert!(validate(&cfg).unidentified_blocks.contains(&id(ARM)));
    }
}

#[test]
fn native_prefix_identity_vm_context_backedge_revokes_native_entry_proof() {
    let mut pe = image(false);
    put(&mut pe, EXIT, |a| { a.test(esi, esi).unwrap(); a.je(ARM).unwrap(); a.ret().unwrap(); });
    let cfg = Explorer::new(&pe, STACK).recover_pass_with(
        ENTRY, &HashSet::new(), None, None, None);
    let arm = cfg.block(id(ARM)).unwrap();
    assert!(arm.preds.contains(&id(ENTRY)) && arm.preds.contains(&id(EXIT)));
    assert!(!cfg.native_entry_proofs.contains(&id(ARM)), "the later edge enters with an active context");
    assert!(validate(&cfg).unidentified_blocks.contains(&id(ARM)));
}

#[test]
fn native_prefix_identity_late_unproven_ancestor_revokes_descendants() {
    const OTHER: u64 = IMAGE + 0x1400;
    const JOIN: u64 = IMAGE + 0x1200;
    let mut pe = image(false);
    put(&mut pe, ENTRY, |a| {
        a.test(ecx, ecx).unwrap(); a.je(OTHER).unwrap();
        a.test(esi, esi).unwrap(); a.je(JOIN).unwrap(); a.ret().unwrap();
    });
    put(&mut pe, OTHER, |a| {
        a.pushfq().unwrap(); a.popfq().unwrap();
        a.test(esi, esi).unwrap(); a.je(JOIN).unwrap(); a.ret().unwrap();
    });
    put(&mut pe, JOIN, |a| {
        a.test(r8d, r8d).unwrap(); a.je(ARM).unwrap(); a.ret().unwrap();
    });
    let cfg = Explorer::new(&pe, STACK).recover_pass_with(
        ENTRY, &HashSet::new(), None, None, None);
    assert_eq!(cfg.block(id(JOIN)).unwrap().preds.len(), 2);
    let arm = cfg.block(id(ARM)).unwrap();
    assert_eq!(arm.preds, [id(JOIN)]);
    assert!(matches!(arm.terminator, Terminator::Branch { .. }));
    assert!(!cfg.native_entry_proofs.contains(&id(ARM)), "late unproven history reaches this descendant too");
    assert!(validate(&cfg).unidentified_blocks.contains(&id(ARM)));
}

#[test]
fn native_prefix_identity_proof_survives_expression_cleanup_and_block_reordering() {
    let pe = image(false);
    let mut cfg = Explorer::new(&pe, STACK).recover(ENTRY);
    assert!(cfg.native_entry_proofs.contains(&id(ARM)));
    assert!(cfg.block(id(ARM)).unwrap().from_vm_context);
    cfg.blocks.reverse();
    cfg.simplify_phis();
    cfg.remove_dead_params();
    assert!(cfg.native_entry_proofs.contains(&id(ARM)));
    assert!(cfg.block(id(ARM)).unwrap().from_vm_context);
    assert!(!validate(&cfg).unidentified_blocks.contains(&id(ARM)));
    cfg.native_entry_proofs.clear();
    assert!(validate(&cfg).unidentified_blocks.contains(&id(ARM)),
        "without positive entry proof the original unidentified rule remains active");
}
