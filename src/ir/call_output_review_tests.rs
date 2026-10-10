
use super::*;
use crate::binary::pe::Section;
use iced_x86::{Decoder, DecoderOptions};
use iced_x86::code_asm::*;
const BASE:u64=0x180000000;
const START:u64=BASE+0x1000;
const HELPER:u64=BASE+0x3000;
const STACK:u64=crate::vm::DEFAULT_STACK_BASE;

fn fixture() -> PeFile {
    let mut a=CodeAssembler::new(64).unwrap();
    a.lea(rcx, ptr(rsp+0x20)).unwrap();
    a.call(HELPER).unwrap();
    a.ret().unwrap();
    let code=a.assemble(START).unwrap();
    let mut data=vec![0x90;0x1300];
    data[0x200..0x200+code.len()].copy_from_slice(&code);
    data[0x1200]=0xc3;
    PeFile { data,image_base:BASE,entry_point_rva:0x1000,size_of_image:0x4000,
        sections:vec![
            Section{name:".tvm0".into(),virtual_address:0x1000,virtual_size:0x1000,
                raw_address:0x200,raw_size:0x1000,characteristics:0x60000020},
            Section{name:".text".into(),virtual_address:0x3000,virtual_size:0x100,
                raw_address:0x1200,raw_size:0x100,characteristics:0x60000020}],
        opt_header_offset:0,section_table_offset:0,file_alignment:0x200,
        section_alignment:0x1000,size_of_headers:0x200,loader_bound:vec![] }
}

fn exposed<'a>(pe:&'a PeFile) -> Emulator<'a> {
    let mut e=Emulator::new(pe,STACK);
    assert!(matches!(e.run(START,2),Stop::Budget{..}));
    assert!(e.guest_call_output_slots.contains(&(STACK+0x20)));
    e
}

#[test]
fn out_review_symbolic_store_expires_future_slot_origin() {
    let pe=fixture(); let mut e=exposed(&pe);
    let where_=e.arena.init_reg(Reg::Rbx);
    let value=e.arena.opaque("scalar_after_call",Width::W64);
    e.store(where_,value,Width::W64,START+16);
    assert!(e.guest_call_output_slots.is_empty(),
        "a write through unknown RBX may overlap the former output qword");
}

#[test]
fn out_review_unknown_boxed_write_expires_future_slot_origin() {
    let pe=fixture(); let mut e=exposed(&pe);
    let where_=e.arena.init_reg(Reg::Rbx);
    e.forget_memory(where_,16);
    assert!(e.guest_call_output_slots.is_empty(),
        "unknown 16-byte boxed write may overlap the former output qword");
}

#[test]
fn out_review_rep_stosq_separates_concrete_load_generations() {
    let pe=fixture(); let mut e=exposed(&pe);
    let slot=e.arena.constant(STACK+0x20,Width::W64);
    let before=e.load(slot,Width::W64,START+16);
    e.state.set_reg(Reg::Rdi,slot);
    let one=e.arena.constant(1,Width::W64); e.state.set_reg(Reg::Rcx,one);
    let seven=e.arena.constant(7,Width::W64); e.state.set_reg(Reg::Rax,seven);
    let mut d=Decoder::with_ip(64,&[0xf3,0x48,0xab],START+20,DecoderOptions::NONE);
    let i=d.decode();
    assert_eq!(i.mnemonic(),Mnemonic::Stosq);
    e.box_instruction(&i,i.ip(),i.next_ip());
    let after=e.load(slot,Width::W64,START+24);
    assert_ne!(before,after,
        "REP STOSQ writes seven to the output slot; a fresh read must not reuse the pre-write Load");
    assert!(e.guest_call_output_slots.is_empty(),
        "the former output location now contains a scalar");
}

#[test]
fn out_review_unrelated_concrete_store_retains_and_partial_overlap_expires() {
    let pe=fixture(); let mut e=exposed(&pe);
    let no_overlap=e.arena.constant(STACK+0x28,Width::W64);
    let v=e.arena.constant(7,Width::W8);
    e.store(no_overlap,v,Width::W8,START+16);
    assert!(e.guest_call_output_slots.contains(&(STACK+0x20)));
    let overlap=e.arena.constant(STACK+0x27,Width::W64);
    e.store(overlap,v,Width::W8,START+17);
    assert!(e.guest_call_output_slots.is_empty());
}

#[test]
fn out_review_dynamic_xsave_write_invalidates_known_bytes_and_output_origin() {
    for bytes in [
        &[0x48, 0x0f, 0xae, 0x20][..], // XSAVE64 [RAX]
        &[0x48, 0x0f, 0xae, 0x30][..], // XSAVEOPT64 [RAX]
        &[0x48, 0x0f, 0xc7, 0x20][..], // XSAVEC64 [RAX]
        &[0x48, 0x0f, 0xc7, 0x28][..], // XSAVES64 [RAX]
    ] {
        let mut pe=fixture();
        pe.data[0x220..0x220+bytes.len()].copy_from_slice(bytes);
        let mut e=exposed(&pe);
        let slot=e.arena.constant(STACK+0x20,Width::W64);
        let before=e.load(slot,Width::W64,START+16);
        let stale=e.arena.constant(0x1111222233334444,Width::W64);
        e.state.store_concrete(&e.arena,STACK+0x100,stale,Width::W64);
        e.state.set_reg(Reg::Rax,slot);
        let mut d=Decoder::with_ip(64,bytes,START+0x20,DecoderOptions::NONE);
        let inst=d.decode();
        assert_eq!(inst.op0_kind(),OpKind::Memory);
        assert_eq!(inst.memory_size().size(),0);
        assert!(matches!(e.run(START+0x20,1),Stop::Budget{..}));
        assert!(e.events.iter().any(|event|matches!(event,Event::Boxed{bytes:raw,..} if raw==bytes)),
            "the actual XSAVE instruction must pass through the boxing path");
        let after=e.load(slot,Width::W64,START+0x24);
        assert_ne!(before,after,"dynamic XSAVE memory write needs a fresh generation");
        assert!(e.guest_call_output_slots.is_empty(),"dynamic extent may overlap the output slot");
        let other=e.arena.constant(STACK+0x100,Width::W64);
        let observed=e.load(other,Width::W64,START+0x24);
        assert!(e.arena.as_const(observed).is_none(),"dynamic extent may overwrite this known concrete value");
    }
}

#[test]
fn out_review_fixed_fxsave_write_keeps_nonoverlapping_known_bytes() {
    let bytes=&[0x48,0x0f,0xae,0x00]; // FXSAVE64 [RAX], fixed 512 bytes
    let mut pe=fixture();
    pe.data[0x220..0x220+bytes.len()].copy_from_slice(bytes);
    let mut e=exposed(&pe);
    let slot=e.arena.constant(STACK+0x20,Width::W64);
    let stale=e.arena.constant(0x1111222233334444,Width::W64);
    e.state.store_concrete(&e.arena,STACK+0x100,stale,Width::W64);
    e.state.store_concrete(&e.arena,STACK+0x220,stale,Width::W64);
    e.state.set_reg(Reg::Rax,slot);
    let mut d=Decoder::with_ip(64,bytes,START+0x20,DecoderOptions::NONE);
    assert_eq!(d.decode().memory_size().size(),512);
    assert!(matches!(e.run(START+0x20,1),Stop::Budget{..}));
    assert!(e.guest_call_output_slots.is_empty());
    let inside=e.arena.constant(STACK+0x100,Width::W64);
    let outside=e.arena.constant(STACK+0x220,Width::W64);
    let changed=e.load(inside,Width::W64,START+0x24);
    let unchanged=e.load(outside,Width::W64,START+0x24);
    assert!(e.arena.as_const(changed).is_none());
    assert_eq!(e.arena.as_const(unchanged),Some(0x1111222233334444),
        "the first qword outside the fixed FXSAVE area must retain its known value");
}
