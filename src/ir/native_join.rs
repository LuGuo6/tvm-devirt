//! Native machine-state joins. Physical-register homes require a positive
//! complete-history proof; failure is never permission to guess a VM image.
use super::*;
use crate::ir::expr::{BinOp, BlockRef, GPRS, UnOp};
use crate::vm::state::{Byte, FLAGS};
use std::collections::BTreeMap;

type ProofResult<T> = Result<T, String>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum PureOp {
    Const(u64), Init(Reg), Bin(BinOp, usize, usize), Un(UnOp, usize),
    Zext(usize), Sext(usize), Trunc(usize), Select(usize, usize, usize),
}

#[derive(Default)]
struct PureInterner { nodes: HashMap<(Width, PureOp), usize> }

impl PureInterner {
    fn key(&mut self, arena: &Arena, value: Ref, memo: &mut HashMap<Ref,usize>, depth: usize) -> ProofResult<usize> {
        if let Some(&key) = memo.get(&value) { return Ok(key); }
        if depth > 128 || self.nodes.len() > 100_000 { return Err("native state proof budget exhausted".into()); }
        let op = match *arena.op(value) {
            Op::Const(c) => PureOp::Const(c), Op::InitReg(r) => PureOp::Init(r),
            Op::Bin(o,a,b) => PureOp::Bin(o,self.key(arena,a,memo,depth+1)?,self.key(arena,b,memo,depth+1)?),
            Op::Un(o,a) => PureOp::Un(o,self.key(arena,a,memo,depth+1)?),
            Op::Zext(a) => PureOp::Zext(self.key(arena,a,memo,depth+1)?),
            Op::Sext(a) => PureOp::Sext(self.key(arena,a,memo,depth+1)?),
            Op::Trunc(a) => PureOp::Trunc(self.key(arena,a,memo,depth+1)?),
            Op::Select(c,a,b) => PureOp::Select(self.key(arena,c,memo,depth+1)?,self.key(arena,a,memo,depth+1)?,self.key(arena,b,memo,depth+1)?),
            Op::Opaque(..) | Op::Load(..) | Op::Param(..) => return Err(format!("unmapped dynamic native state {}", crate::ir::expr::render(arena,value,2))),
        };
        let count = self.nodes.len();
        let key = *self.nodes.entry((arena.width(value),op)).or_insert(count);
        memo.insert(value,key); Ok(key)
    }
}

#[derive(Debug, PartialEq, Eq)]
enum ByteKey { Const(u8), Pure(usize,u32) }

#[derive(Debug, PartialEq, Eq)]
struct Signature {
    memory: BTreeMap<u64,ByteKey>, flags: BTreeMap<u8,usize>,
    xmm: BTreeMap<(u8,u8),usize>, written_args: Vec<u8>,
    call_result_homes: Vec<Vec<u8>>, call_output_slots: Vec<u64>,
    call_clobbered_gprs: Vec<u8>,
}

pub(crate) struct NativeJoinPlan {
    rsp: u64, live_flags: u32, signature: Option<Signature>, interner: PureInterner,
}

fn flag_bit(f: Flag) -> u32 {
    use iced_x86::RflagsBits as F;
    match f { Flag::Cf=>F::CF, Flag::Pf=>F::PF, Flag::Af=>F::AF, Flag::Zf=>F::ZF,
        Flag::Sf=>F::SF, Flag::Of=>F::OF, Flag::Df=>F::DF }
}

/// A flag is dead only when every reachable real instruction overwrites it
/// before reading it, or returns (flags are not Windows ABI return values).
/// Unknown/indirect control flow keeps all remaining flags live.
fn live_flags(pe: &PeFile, resume: u64) -> u32 {
    use iced_x86::FlowControl as F;
    let all = FLAGS.iter().fold(0, |m,&f| m | flag_bit(f));
    let mut queue = vec![(resume,all)];
    let mut seen = HashSet::new();
    let mut live = 0;
    while let Some((ip,mut remaining)) = queue.pop() {
        remaining &= !live;
        if remaining == 0 || !seen.insert((ip,remaining)) { continue; }
        if seen.len() > 4096 { return all; }
        let Some(inst) = disasm::decode_at(pe,ip) else { live |= remaining; continue; };
        live |= remaining & inst.rflags_read();
        remaining &= !inst.rflags_modified();
        remaining &= !live;
        if remaining == 0 { continue; }
        match inst.flow_control() {
            F::Return => {},
            F::Next => queue.push((inst.next_ip(),remaining)),
            F::ConditionalBranch => {
                if let Some(target) = disasm::direct_target(&inst) { queue.push((target,remaining)); }
                else { live |= remaining; }
                queue.push((inst.next_ip(),remaining));
            },
            F::UnconditionalBranch => {
                if let Some(target) = disasm::direct_target(&inst) { queue.push((target,remaining)); }
                else { live |= remaining; }
            },
            _ => live |= remaining,
        }
    }
    live
}

impl NativeJoinPlan {
    fn contains_call_clobber(emu: &Emulator<'_>, root: Ref) -> bool {
        Self::contains_uncaptured_call_output(&emu.arena, root)
    }

    pub(crate) fn contains_uncaptured_call_output(arena: &Arena, root: Ref) -> bool {
        let mut pending = vec![root];
        let mut seen = HashSet::new();
        while let Some(value) = pending.pop() {
            if !seen.insert(value) { continue; }
            if seen.len() > 100_000 { return true; }
            if matches!(arena.op(value), Op::Opaque("clobbered" | "flag", _)) {
                return true;
            }
            pending.extend(arena.children(value));
        }
        false
    }

    fn reads_call_clobber(emu: &Emulator<'_>, inst: &Instruction) -> bool {
        // A zero idiom does not consume its former architectural value.
        let zero_idiom = matches!(inst.mnemonic(), Mnemonic::Xor | Mnemonic::Sub)
            && inst.op_count() == 2 && inst.op_kind(0) == OpKind::Register
            && inst.op_kind(1) == OpKind::Register && inst.op_register(0) == inst.op_register(1);
        if !zero_idiom {
            let mut factory = InstructionInfoFactory::new();
            for used in factory.info(inst).used_registers() {
                if matches!(used.access(), OpAccess::Read | OpAccess::CondRead
                    | OpAccess::ReadWrite | OpAccess::ReadCondWrite)
                    && let Some((reg, _, _)) = decode_reg(used.register())
                    && Self::contains_call_clobber(emu, emu.state.reg(reg)) {
                    return true;
                }
            }
        }
        FLAGS.iter().any(|&flag| inst.rflags_read() & flag_bit(flag) != 0
            && Self::contains_call_clobber(emu, emu.state.flag(flag)))
    }

    /// A call origin can move to a block parameter only when its complete value
    /// is held by that exact physical register on every incoming edge. Nested
    /// expressions are not substitutes for the origin they contain. Memory,
    /// live flags and XMM are independently required to remain pure/identical.
    fn call_result_homes(emu: &Emulator<'_>) -> ProofResult<Vec<Vec<u8>>> {
        let mut by_origin: HashMap<Ref, Vec<u8>> = HashMap::new();
        for &reg in GPRS.iter().filter(|&&r| r != Reg::Rsp) {
            let value = emu.state.reg(reg);
            let mut pending = vec![value];
            let mut seen = HashSet::new();
            while let Some(node) = pending.pop() {
                if !seen.insert(node) { continue; }
                if seen.len() > 100_000 {
                    return Err("native call-origin proof budget exhausted".into());
                }
                if emu.guest_call_results.contains(&node) {
                    if node != value {
                        return Err("native call origin is nested inside a seeded register".into());
                    }
                    by_origin.entry(node).or_default().push(reg as u8);
                }
                pending.extend(emu.arena.children(node));
            }
        }
        let mut normalized: Vec<_> = by_origin.into_values().collect();
        for homes in &mut normalized { homes.sort_unstable(); homes.dedup(); }
        normalized.sort_unstable();
        Ok(normalized)
    }

    fn machine_mode(emu: &Emulator<'_>) -> bool {
        emu.guest_ctx.is_none() && emu.guest_layout.is_none()
            && emu.vip_slot.is_none() && emu.unwind_slot.is_none()
            && !emu.diagnostic_context_origin && !emu.diagnostic_location_homes
            && !emu.context_tracker.required
            && std::env::var_os("TVM_DIAGNOSTIC_INLINE_CHKSTK").is_none()
    }

    /// A pre-discovered slot is only a locator. It does not establish that the
    /// function has entered VM mode before the slot has any actual contents.
    /// This proof never authorizes native register homes or parameter seeding.
    fn prefix_mode(emu: &Emulator<'_>) -> bool {
        emu.guest_ctx.is_none() && emu.guest_layout.is_none()
            && emu.current_vip().is_none() && emu.unwind_slot.is_none()
            && !emu.diagnostic_context_origin && !emu.diagnostic_location_homes
            && !emu.context_tracker.required
            && std::env::var_os("TVM_DIAGNOSTIC_INLINE_CHKSTK").is_none()
    }

    pub(crate) fn prefix_history(emu: &Emulator<'_>) -> bool {
        emu.native_prefix_complete && emu.native_history_steps == emu.steps
            && Self::prefix_mode(emu)
    }

    /// Positive instruction gate, independent of the section name. The basic
    /// evaluator also supports approximated arithmetic flags, TVM exception
    /// exits and boxed instructions; those are not a physical-state proof.
    pub(crate) fn record_step(emu: &mut Emulator<'_>, inst: &Instruction) {
        use Mnemonic as M;
        let supported = match inst.mnemonic() {
            M::Nop | M::Pause | M::Mov | M::Movzx | M::Movsx | M::Movsxd | M::Lea
            | M::Add | M::Sub | M::Cmp | M::Test | M::And | M::Or | M::Xor
            | M::Not | M::Neg | M::Inc | M::Dec | M::Bswap
            | M::Cdq | M::Cqo | M::Cwd
            | M::Jmp | M::Call | M::Ret => true,
            M::Push => inst.stack_pointer_increment() == -8,
            M::Pop => inst.stack_pointer_increment() == 8,
            M::Xchg => (0..inst.op_count()).all(|n| inst.op_kind(n) != OpKind::Memory),
            other => cmov_cc(other).is_some() || setcc_cc(other).is_some() || jcc_cc(other).is_some(),
        };
        let registers_supported = (0..inst.op_count()).all(|n|
            inst.op_kind(n) != OpKind::Register || decode_reg(inst.op_register(n)).is_some());
        // mem_address models full 64-bit base/index arithmetic. An address-size
        // override needs a final 32-bit truncation that it does not yet perform.
        let address_supported = [inst.memory_base(), inst.memory_index()].iter().all(|&r|
            r == Register::None || r == Register::RIP
                || decode_reg(r).is_some_and(|(_, offset, width)| offset == 0 && width == Width::W64));
        let step_is_complete = emu.native_history_steps.checked_add(1) == Some(emu.steps)
            && supported && registers_supported && address_supported
            && !inst.has_lock_prefix() && !seg_relative(inst)
            && !Self::reads_call_clobber(emu, inst);
        emu.native_prefix_complete = emu.native_prefix_complete
            && step_is_complete && Self::prefix_mode(emu);
        emu.native_effects_complete = emu.native_effects_complete
            && step_is_complete && Self::machine_mode(emu);
        emu.native_history_steps = emu.steps;
    }

    fn history(emu: &Emulator<'_>) -> bool {
        emu.native_effects_complete && emu.native_history_steps == emu.steps
            && Self::machine_mode(emu)
    }

    /// Called before a block's events are drained. The sticky history bit stops
    /// a later context-free block from hiding an earlier discarded effect.
    pub(crate) fn snapshot(emu: &mut Emulator<'_>) -> Option<Vec<(Reg, Ref)>> {
        let all_effects = emu.events.iter().all(|event| match event {
            Event::Load { region, .. } | Event::Store { region, .. } => region.is_guest(),
            Event::StackProbe { .. } | Event::Boxed { .. } => false,
            _ => true,
        });
        emu.native_prefix_complete &= all_effects && Self::prefix_history(emu);
        emu.native_effects_complete &= all_effects && Self::history(emu);
        Self::history(emu).then(|| GPRS.iter().map(|&r| (r,emu.state.reg(r))).collect())
    }

    fn frame(emu: &Emulator<'_>, homes: &[(Reg,Ref)]) -> ProofResult<u64> {
        if !Self::history(emu) { return Err("native architectural history is not proven complete".into()); }
        if homes.len()!=GPRS.len() || GPRS.iter().any(|r|
            homes.iter().filter(|(reg,_)| reg==r).count()!=1
            || !homes.iter().any(|(reg,v)| reg==r && emu.state.reg(*r)==*v)) {
            return Err("native incoming homes differ from exact parent machine registers".into());
        }
        let rsp = emu.state.concrete_rsp(&emu.arena).ok_or("native incoming RSP is not concrete")?;
        if rsp.abs_diff(emu.stack_base)>crate::ir::sched::FRAME_WINDOW { return Err("native RSP outside frame window".into()); }
        Ok(rsp)
    }

    pub(crate) fn discover(emu: &Emulator<'_>, homes: &[(Reg,Ref)], resume: u64) -> ProofResult<Self> {
        let mut plan = Self { rsp:Self::frame(emu,homes)?, live_flags:live_flags(emu.pe,resume),
            signature:None, interner:PureInterner::default() };
        plan.validate(emu,homes)?;
        Ok(plan)
    }

    pub(crate) fn validate(&mut self, emu: &Emulator<'_>, homes: &[(Reg,Ref)]) -> ProofResult<()> {
        if Self::frame(emu,homes)?!=self.rsp { return Err("native incoming RSP differs".into()); }
        if !emu.state.sym_stores.is_empty() || emu.state.has_symbolic_forwarding() {
            return Err("native join has unmapped symbolic memory".into());
        }
        let mut memo = HashMap::new();
        let mut memory = BTreeMap::new();
        for (&address,byte) in &emu.state.mem {
            let key = match *byte {
                Byte::Const(b) => ByteKey::Const(b),
                Byte::Sym {expr,index} => ByteKey::Pure(self.interner.key(&emu.arena,expr,&mut memo,0)?,index),
            };
            memory.insert(address,key);
        }
        let mut flags = BTreeMap::new();
        for &flag in &FLAGS {
            if self.live_flags & flag_bit(flag) != 0 {
                flags.insert(flag as u8,self.interner.key(&emu.arena,emu.state.flag(flag),&mut memo,0)?);
            }
        }
        let mut xmm = BTreeMap::new();
        for (&(r,h),&v) in &emu.state.xmm { xmm.insert((r,h as u8),self.interner.key(&emu.arena,v,&mut memo,0)?); }
        let mut written_args:Vec<_> = emu.written_args.iter().map(|r| *r as u8).collect();
        written_args.sort_unstable();
        let call_result_homes = Self::call_result_homes(emu)?;
        let mut call_output_slots: Vec<_> = emu.guest_call_output_slots.iter().copied().collect();
        call_output_slots.sort_unstable();
        // A later incoming edge may carry an uncaptured call output even when
        // the first edge had a pure register value. Such a value must not reuse
        // the first edge's already-readable parameter.
        let call_clobbered_gprs = GPRS.iter().filter(|&&reg| reg != Reg::Rsp
            && Self::contains_call_clobber(emu, emu.state.reg(reg)))
            .map(|&reg| reg as u8).collect();
        let signature = Signature { memory,flags,xmm,written_args,call_result_homes,
            call_output_slots,call_clobbered_gprs };
        if self.signature.as_ref().is_some_and(|old| old!=&signature) {
            return Err("unparameterized native memory, live flags, XMM, call arguments or call origins differ".into());
        }
        self.signature = Some(signature); Ok(())
    }

    pub(crate) fn seed(&self, emu: &mut Emulator<'_>, block: BlockRef) -> Vec<(Reg,Ref)> {
        // Pins came from individual incoming paths. A join widens them; keeping
        // them after replacing register values would specialize the new block.
        emu.pins.clear(); emu.subst_memo.clear(); emu.pin_dep.clear(); emu.memo_pin_count=0;
        emu.state.forget_symbolic_memory();
        emu.arena.bump_mem_gen(); emu.arena.note_symbolic_store();
        emu.guest_call_results.clear();
        let call_homes = &self.signature.as_ref().expect("validated native join").call_result_homes;
        for &flag in &FLAGS {
            if self.live_flags & flag_bit(flag)==0 {
                let unused = emu.arena.constant(0,Width::W8);
                emu.state.set_flag(flag,unused);
            }
        }
        let mut seeded = Vec::new();
        for &reg in GPRS.iter().filter(|&&r| r!=Reg::Rsp) {
            // The ABI call model captures RAX only. Do not turn an uncaptured
            // volatile output into a seemingly valid parameter. Retain its
            // taint so any later read permanently revokes machine proof.
            if Self::contains_call_clobber(emu, emu.state.reg(reg)) { continue; }
            let value = emu.arena.param(block,reg);
            if call_homes.iter().any(|homes| homes.contains(&(reg as u8))) {
                emu.guest_call_results.insert(value);
            }
            emu.state.set_reg(reg,value);
            seeded.push((reg,value));
        }
        seeded
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binary::pe::Section;
    use iced_x86::{code_asm::*, RflagsBits as F};

    const IMAGE: u64 = 0x180000000;
    const ENTRY: u64 = IMAGE + 0x1000;
    const STACK: u64 = 0x7fff00003000;

    fn fixture(emit: impl FnOnce(&mut CodeAssembler)) -> PeFile {
        let mut a = CodeAssembler::new(64).unwrap();
        emit(&mut a);
        let bytes = a.assemble(ENTRY).unwrap();
        let mut data = vec![0xcc; 0x1200];
        data[0x200..0x200+bytes.len()].copy_from_slice(&bytes);
        PeFile { data, image_base: IMAGE, entry_point_rva: 0x1000,
            size_of_image: 0x3000,
            sections: vec![Section { name: ".text".into(), virtual_address: 0x1000,
                virtual_size: 0x1000, raw_address: 0x200, raw_size: 0x1000,
                characteristics: 0x60000020 }],
            opt_header_offset: 0, section_table_offset: 0, file_alignment: 0x200,
            section_alignment: 0x1000, size_of_headers: 0x200, loader_bound: vec![] }
    }

    fn homes(emu: &Emulator<'_>) -> Vec<(Reg, Ref)> {
        GPRS.iter().map(|&r| (r, emu.state.reg(r))).collect()
    }

    fn all_flags() -> u32 { FLAGS.iter().fold(0, |m, &f| m | flag_bit(f)) }

    #[test]
    fn native_join_pushfq_reads_flags_before_a_later_overwrite() {
        let pe = fixture(|a| {
            a.pushfq().unwrap(); a.pop(rax).unwrap();
            a.xor(ecx, ecx).unwrap(); a.ret().unwrap();
        });
        assert_eq!(live_flags(&pe, ENTRY), all_flags());
        let mut left = Emulator::new(&pe, STACK);
        for f in FLAGS {
            let c = left.arena.constant(0, Width::W8);
            left.state.set_flag(f, c);
        }
        let mut right = left.clone();
        let one = right.arena.constant(1, Width::W8);
        right.state.set_flag(Flag::Cf, one);
        let mut plan = NativeJoinPlan::discover(&left, &homes(&left), ENTRY).unwrap();
        assert!(plan.validate(&right, &homes(&right)).is_err());
    }

    #[test]
    fn native_join_call_before_return_keeps_incoming_flags_live() {
        let pe = fixture(|a| { a.call(ENTRY + 0x100).unwrap(); a.ret().unwrap(); });
        assert_eq!(live_flags(&pe, ENTRY), all_flags(),
            "a helper may consume flags even though the caller later returns");
    }

    #[test]
    fn native_join_adc_reads_cf_before_overwriting_it() {
        let pe = fixture(|a| { a.adc(eax, 0).unwrap(); a.ret().unwrap(); });
        assert_eq!(live_flags(&pe, ENTRY), F::CF);
    }

    #[test]
    fn native_join_undefined_flags_are_written_by_the_instruction() {
        let pe = fixture(|a| { a.xor(eax, eax).unwrap(); a.lahf().unwrap(); a.ret().unwrap(); });
        let inst = disasm::decode_at(&pe, ENTRY).unwrap();
        assert_ne!(inst.rflags_undefined() & F::AF, 0);
        assert_ne!(inst.rflags_modified() & F::AF, 0);
        assert_eq!(live_flags(&pe, ENTRY), 0,
            "LAHF reads the XOR result, including its undefined AF, not the incoming AF");
        let mut emu = Emulator::new(&pe, STACK);
        let plan = NativeJoinPlan::discover(&emu, &homes(&emu), ENTRY).unwrap();
        plan.seed(&mut emu, BlockRef(123));
        emu.step(ENTRY);
        assert!(matches!(emu.arena.op(emu.state.flag(Flag::Af)), Op::Opaque(..)),
            "the real instruction's undefined flag must remain unknown after seeding");
    }

    #[test]
    fn native_join_rejects_late_rsp_and_unparameterized_memory_changes() {
        let pe = fixture(|a| { a.ret().unwrap(); });
        let left = Emulator::new(&pe, STACK);
        let mut plan = NativeJoinPlan::discover(&left, &homes(&left), ENTRY).unwrap();
        let mut right = left.clone();
        let stack_ref = right.arena.constant(STACK - 8, Width::W64);
        right.state.set_reg(Reg::Rsp, stack_ref);
        assert!(plan.validate(&right, &homes(&right)).unwrap_err().contains("RSP differs"));
        let mut right = left.clone();
        right.state.mem.insert(STACK + 0x28, Byte::Const(1));
        assert!(plan.validate(&right, &homes(&right)).is_err());
    }

    #[test]
    fn native_join_rejects_same_address_different_bytes_and_opaque_id_collisions() {
        let pe = fixture(|a| { a.ret().unwrap(); });
        let mut left = Emulator::new(&pe, STACK);
        left.state.mem.insert(STACK + 0x28, Byte::Const(0));
        let mut right = left.clone();
        right.state.mem.insert(STACK + 0x28, Byte::Const(1));
        let mut plan = NativeJoinPlan::discover(&left, &homes(&left), ENTRY).unwrap();
        assert!(plan.validate(&right, &homes(&right)).is_err());
        let dynamic = right.arena.opaque("different_arena_history", Width::W64);
        right.state.mem.insert(STACK + 0x28, Byte::Sym { expr: dynamic, index: 0 });
        assert!(NativeJoinPlan::discover(&right, &homes(&right), ENTRY).is_err());
    }

    #[test]
    fn native_join_history_rejects_context_vip_and_untracked_steps() {
        let pe = fixture(|a| { a.nop().unwrap(); a.ret().unwrap(); });
        let mut clean = Emulator::new(&pe, STACK);
        clean.step(ENTRY);
        assert!(NativeJoinPlan::snapshot(&mut clean).is_some());
        let mut with_context = clean.clone();
        with_context.guest_ctx = Some(STACK - 0x100);
        assert!(NativeJoinPlan::snapshot(&mut with_context).is_none());
        let mut with_vip = clean.clone();
        with_vip.vip_slot = Some(0);
        assert!(NativeJoinPlan::snapshot(&mut with_vip).is_none());
        let mut truncated = clean.clone();
        truncated.steps += 1;
        assert!(NativeJoinPlan::snapshot(&mut truncated).is_none());
    }

    #[test]
    fn native_join_complete_machine_steps_can_be_in_the_vm_section() {
        let mut pe = fixture(|a| { a.nop().unwrap(); a.ret().unwrap(); });
        pe.sections[0].name = ".tvm0".into();
        let mut emu = Emulator::new(&pe, STACK);
        assert!(emu.in_vm(ENTRY));
        emu.step(ENTRY);
        assert_eq!(emu.native_history_steps, 1);
        assert!(NativeJoinPlan::snapshot(&mut emu).is_some());
        // Merely lacking a context is insufficient: a TVM-specific instruction
        // in the same section must permanently revoke the proof.
        let mut pe = fixture(|a| { a.rdsspq(rax).unwrap(); a.ret().unwrap(); });
        pe.sections[0].name = ".tvm0".into();
        let mut emu = Emulator::new(&pe, STACK);
        emu.step(ENTRY);
        assert!(emu.guest_ctx.is_none());
        assert!(NativeJoinPlan::snapshot(&mut emu).is_none());
    }

    #[test]
    fn native_join_complete_history_survives_trace_tail_eviction() {
        let pe = fixture(|a| { a.nop().unwrap(); a.jmp(ENTRY).unwrap(); });
        let mut emu = Emulator::new(&pe, STACK);
        emu.stop_on_revisit = false;
        emu.run(ENTRY, 5000);
        assert_eq!(emu.steps, 5000);
        assert_eq!(emu.native_history_steps, 5000);
        assert_eq!(emu.trace.len(), 4096);
        assert!(NativeJoinPlan::snapshot(&mut emu).is_some());
        emu.native_history_steps -= 1;
        assert!(NativeJoinPlan::snapshot(&mut emu).is_none());
    }

    #[test]
    fn native_join_rejects_approximated_arithmetic_and_boxed_effects() {
        for bytes in [vec![0x48, 0xd1, 0xe0], vec![0x0f, 0xa2], vec![0x48, 0x11, 0xc8]] {
            let mut pe = fixture(|a| { a.nop().unwrap(); });
            pe.data[0x200..0x200+bytes.len()].copy_from_slice(&bytes);
            let mut emu = Emulator::new(&pe, STACK);
            emu.step(ENTRY);
            assert!(NativeJoinPlan::snapshot(&mut emu).is_none());
        }
    }

    #[test]
    fn native_join_rejects_unmodeled_registers_address_wrap_and_full_rflags() {
        for bytes in [
            vec![0x8e, 0xd8], // mov ds, ax: decode_reg intentionally has no DS.
            vec![0x0f, 0x22, 0xc0], // mov cr0, rax: privileged state is not a GPR.
            vec![0x67, 0x8b, 0x04, 0x11], // mov eax, [ecx+edx]: 32-bit address wrap.
            vec![0x9c], // pushfq: the evaluator models only seven RFLAGS bits.
            vec![0x9d], // popfq: IF/IOPL and other architectural bits are not modeled.
        ] {
            let mut pe = fixture(|a| { a.nop().unwrap(); });
            pe.data[0x200..0x200+bytes.len()].copy_from_slice(&bytes);
            let mut emu = Emulator::new(&pe, STACK);
            emu.step(ENTRY);
            assert!(NativeJoinPlan::snapshot(&mut emu).is_none(), "accepted {bytes:02x?}");
        }
    }

    #[test]
    fn native_join_does_not_assume_runtime_bss_is_still_zero() {
        let mut pe = fixture(|a| {
            a.mov(r11, IMAGE + 0x2000).unwrap(); a.mov(eax, dword_ptr(r11)).unwrap(); a.ret().unwrap();
        });
        pe.sections.push(Section { name: ".bss".into(), virtual_address: 0x2000,
            virtual_size: 0x1000, raw_address: 0x1200, raw_size: 0,
            characteristics: 0xc0000080 });
        let mut emu = Emulator::new(&pe, STACK);
        emu.run(ENTRY, 2);
        assert!(emu.pe.is_bss(IMAGE + 0x2000));
        assert!(NativeJoinPlan::snapshot(&mut emu).is_none());
    }

    #[test]
    fn native_join_discarded_effect_rejection_is_sticky_after_event_drain() {
        let pe = fixture(|a| { a.nop().unwrap(); a.ret().unwrap(); });
        let mut emu = Emulator::new(&pe, STACK);
        emu.step(ENTRY);
        let address = emu.arena.constant(STACK - 0x300, Width::W64);
        let value = emu.arena.constant(1, Width::W64);
        emu.events.push(Event::Store { addr: address, value, width: Width::W64,
            site: ENTRY, region: Region::VmScratch });
        assert!(NativeJoinPlan::snapshot(&mut emu).is_none());
        emu.events.clear();
        assert!(NativeJoinPlan::snapshot(&mut emu).is_none());
    }

    fn call_state(pe: &PeFile) -> Emulator<'_> {
        let mut emu = Emulator::new(pe, STACK);
        let origin = emu.arena.opaque("call_ret", Width::W64);
        emu.guest_call_results.insert(origin);
        emu.state.set_reg(Reg::R10, origin);
        emu
    }

    #[test]
    fn native_join_call_origin_seeding_uses_new_parameter_and_drops_dead_origins() {
        let pe = fixture(|a| { a.ret().unwrap(); });
        let mut emu = call_state(&pe);
        let dead = emu.arena.opaque("call_ret", Width::W64);
        emu.guest_call_results.insert(dead);
        let homes = homes(&emu);
        let plan = NativeJoinPlan::discover(&emu, &homes, ENTRY).unwrap();
        let block = BlockRef(1);
        let seeded = plan.seed(&mut emu, block);
        let seeded_r10 = seeded.iter().find(|(r, _)| *r == Reg::R10).unwrap().1;
        assert_eq!(emu.guest_call_results, HashSet::from([seeded_r10]));
        assert_ne!(seeded_r10, dead);
    }

    #[test]
    fn native_join_call_origin_nested_uses_and_unmapped_memory_stay_rejected() {
        let pe = fixture(|a| { a.ret().unwrap(); });
        let mut emu = call_state(&pe);
        let origin = emu.state.reg(Reg::R10);
        let four = emu.arena.constant(4, Width::W64);
        let nested = emu.arena.bin(BinOp::Add, origin, four);
        emu.state.set_reg(Reg::R11, nested);
        assert!(NativeJoinPlan::discover(&emu, &homes(&emu), ENTRY).is_err());
        emu.state.set_reg(Reg::R11, four);
        emu.state.store_concrete(&emu.arena, STACK-8, origin, Width::W64);
        assert!(NativeJoinPlan::discover(&emu, &homes(&emu), ENTRY).is_err());
    }

    #[test]
    fn native_join_call_origin_home_aliases_and_output_slots_need_all_incoming_match() {
        let pe = fixture(|a| { a.ret().unwrap(); });
        let left = call_state(&pe);
        let mut plan = NativeJoinPlan::discover(&left, &homes(&left), ENTRY).unwrap();
        let mut right = call_state(&pe);
        assert!(plan.validate(&right, &homes(&right)).is_ok());
        right.state.set_reg(Reg::R11, right.state.reg(Reg::R10));
        assert!(plan.validate(&right, &homes(&right)).is_err());
        let mut right = call_state(&pe);
        right.guest_call_output_slots.insert(STACK-0x40);
        assert!(plan.validate(&right, &homes(&right)).is_err());
    }

    #[test]
    fn native_join_call_origin_aliases_are_both_reseeded_and_removal_is_rejected() {
        let pe = fixture(|a| { a.ret().unwrap(); });
        let mut left = call_state(&pe);
        left.state.set_reg(Reg::R11, left.state.reg(Reg::R10));
        let mut plan = NativeJoinPlan::discover(&left, &homes(&left), ENTRY).unwrap();
        let mut right = call_state(&pe);
        right.state.set_reg(Reg::R11, right.state.reg(Reg::R10));
        assert!(plan.validate(&right, &homes(&right)).is_ok());
        let mut missing_alias = right.clone();
        let zero = missing_alias.arena.constant(0, Width::W64);
        missing_alias.state.set_reg(Reg::R11, zero);
        assert!(plan.validate(&missing_alias, &homes(&missing_alias)).is_err());
        plan.seed(&mut right, BlockRef(2));
        assert_eq!(right.guest_call_results,
            HashSet::from([right.state.reg(Reg::R10), right.state.reg(Reg::R11)]));
    }

    #[test]
    fn native_join_uncaptured_call_outputs_reject_reads_but_allow_overwrites() {
        let pe = fixture(|a| { a.test(edx, edx).unwrap(); a.ret().unwrap(); });
        let mut emu = call_state(&pe);
        let unknown = emu.arena.opaque("clobbered", Width::W64);
        emu.state.set_reg(Reg::Rdx, unknown);
        let plan = NativeJoinPlan::discover(&emu, &homes(&emu), ENTRY).unwrap();
        let params = plan.seed(&mut emu, BlockRef(3));
        assert!(!params.iter().any(|(reg,_)| *reg == Reg::Rdx));
        emu.step(ENTRY);
        assert!(NativeJoinPlan::snapshot(&mut emu).is_none());
        let pe = fixture(|a| { a.xor(edx, edx).unwrap(); a.test(edx, edx).unwrap(); a.ret().unwrap(); });
        let mut emu = call_state(&pe);
        let unknown = emu.arena.opaque("clobbered", Width::W64);
        emu.state.set_reg(Reg::Rdx, unknown);
        emu.run(ENTRY, 2);
        assert!(NativeJoinPlan::snapshot(&mut emu).is_some());
    }
    #[test]
    fn independent_native_late_tainted_incoming_is_not_washed_into_param() {
        let pe = fixture(|a| { a.test(edx, edx).unwrap(); a.ret().unwrap(); });
        let left = call_state(&pe);
        let mut right = left.clone();
        let taint = right.arena.opaque("clobbered", Width::W64);
        right.state.set_reg(Reg::Rdx, taint);
        let mut plan = NativeJoinPlan::discover(&left, &homes(&left), ENTRY).unwrap();
        let mut seeded = left.clone();
        let params = plan.seed(&mut seeded, BlockRef(71));
        assert!(params.iter().any(|(r,_)| *r == Reg::Rdx));
        seeded.run(ENTRY, 1);
        assert!(NativeJoinPlan::snapshot(&mut seeded).is_some());
        assert!(plan.validate(&right, &homes(&right)).is_err(),
            "late incoming has uncaptured RDX but first incoming already seeded readable Param");
    }

    #[test]
    fn independent_native_late_nested_taint_is_not_washed_into_param() {
        let pe = fixture(|a| { a.test(edx, edx).unwrap(); a.ret().unwrap(); });
        let left = call_state(&pe);
        let mut right = left.clone();
        let taint = right.arena.opaque("clobbered", Width::W64);
        let one = right.arena.constant(1, Width::W64);
        let nested = right.arena.bin(BinOp::Add, taint, one);
        right.state.set_reg(Reg::Rdx, nested);
        let mut plan = NativeJoinPlan::discover(&left, &homes(&left), ENTRY).unwrap();
        assert!(plan.validate(&right, &homes(&right)).is_err(),
            "nested uncaptured output must participate in every-incoming signature");
    }

    #[test]
    fn independent_native_same_taint_keeps_rejection_after_seed() {
        let pe = fixture(|a| { a.test(edx, edx).unwrap(); a.ret().unwrap(); });
        let mut left = call_state(&pe);
        let taint = left.arena.opaque("clobbered", Width::W64);
        left.state.set_reg(Reg::Rdx, taint);
        let right = left.clone();
        let mut plan = NativeJoinPlan::discover(&left, &homes(&left), ENTRY).unwrap();
        assert!(plan.validate(&right, &homes(&right)).is_ok());
        let params = plan.seed(&mut left, BlockRef(72));
        assert!(!params.iter().any(|(r,_)| *r == Reg::Rdx));
        left.run(ENTRY, 1);
        assert!(NativeJoinPlan::snapshot(&mut left).is_none());
    }

    #[test]
    fn independent_native_dynamic_call_instances_keep_distinct_homes() {
        let pe = fixture(|a| { a.ret().unwrap(); });
        let mut left = call_state(&pe);
        let new_call = left.arena.opaque("call_ret", Width::W64);
        left.guest_call_results.insert(new_call);
        left.state.set_reg(Reg::R11, new_call);
        let mut right = left.clone();
        let right_old = right.arena.opaque("call_ret", Width::W64);
        let right_new = right.arena.opaque("call_ret", Width::W64);
        right.guest_call_results.extend([right_old,right_new]);
        right.state.set_reg(Reg::R10,right_old);right.state.set_reg(Reg::R11,right_new);
        let mut plan = NativeJoinPlan::discover(&left, &homes(&left), ENTRY).unwrap();
        assert!(plan.validate(&right,&homes(&right)).is_ok());
        plan.seed(&mut right,BlockRef(73));
        assert_ne!(right.state.reg(Reg::R10),right.state.reg(Reg::R11));
        assert!(right.guest_call_results.contains(&right.state.reg(Reg::R10)));
        assert!(right.guest_call_results.contains(&right.state.reg(Reg::R11)));
    }
}
