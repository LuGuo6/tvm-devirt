//! Storage provenance for an executed native restore / return-save round trip.
//! No expression IDs or concrete value equality participate in these proofs.
//! A round trip alone is NOT a current-consumer proof. Production recovery does
//! not use this diagnostic candidate unless explicitly enabled by the caller.
use super::{LAYOUTS, Layout, Reg};
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Transform<T> {
    root: T,
    negated: bool,
    add: u64,
}
impl<T: Copy> Transform<T> {
    fn new(root: T) -> Self {
        Self {
            root,
            negated: false,
            add: 0,
        }
    }
    fn offset(mut self, n: u64) -> Self {
        self.add = self.add.wrapping_add(n);
        self
    }
    fn not(mut self) -> Self {
        self.negated = !self.negated;
        self.add = !self.add;
        self
    }
    fn exact(self) -> Option<T> {
        (!self.negated && self.add == 0).then_some(self.root)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Origin {
    read: Option<Transform<u64>>,
    saved: Option<Transform<(Reg, u64)>>,
}
impl Origin {
    pub fn offset(self, n: u64) -> Self {
        Self {
            read: self.read.map(|x| x.offset(n)),
            saved: self.saved.map(|x| x.offset(n)),
        }
    }
    pub fn not(self) -> Self {
        Self {
            read: self.read.map(Transform::not),
            saved: self.saved.map(Transform::not),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ContextIdentity {
    pub base: u64,
    pub layout_index: usize,
    pub guest_rsp: u64,
}

impl<'a> super::Emulator<'a> {
    fn context_read_operand(
        &self,
        inst: &iced_x86::Instruction,
        op: u32,
        address: Option<u64>,
    ) -> Origin {
        use iced_x86::OpKind;
        match inst.op_kind(op) {
            OpKind::Register => super::decode_reg(inst.op_register(op))
                .filter(|(_, offset, width)| *offset == 0 && *width == super::Width::W64)
                .map(|(r, _, _)| self.context_tracker.reg(r))
                .unwrap_or_default(),
            OpKind::Memory if inst.memory_size().size() == 8 => address
                .map(|a| self.context_tracker.load(a))
                .unwrap_or_default(),
            _ => Origin::default(),
        }
    }

    fn context_write_operand(
        &mut self,
        inst: &iced_x86::Instruction,
        op: u32,
        address: Option<u64>,
        value: Origin,
    ) {
        use iced_x86::OpKind;
        match inst.op_kind(op) {
            OpKind::Register => {
                if let Some((r, offset, width)) = super::decode_reg(inst.op_register(op)) {
                    self.context_tracker.set_reg(
                        r,
                        if offset == 0 && width == super::Width::W64 {
                            value
                        } else {
                            Origin::default()
                        },
                    );
                }
            }
            OpKind::Memory => {
                if let Some(a) = address {
                    self.context_tracker
                        .store(a, inst.memory_size().size() as u32, value, false);
                } else {
                    self.context_tracker.invalidate();
                }
            }
            _ => {}
        }
    }

    pub(super) fn track_context_instruction(&mut self, inst: &iced_x86::Instruction) {
        use iced_x86::{InstructionInfoFactory, Mnemonic as M, OpAccess, OpKind, Register};
        if !self.diagnostic_context_origin && !self.diagnostic_location_homes {
            return;
        }
        if !self.in_vm(inst.ip()) {
            self.context_tracker.invalidate();
            return;
        }
        let memory = (0..inst.op_count()).any(|op| inst.op_kind(op) == OpKind::Memory);
        let address = if memory && !super::seg_relative(inst) {
            let reference = self.mem_address(inst);
            let reference = self.substitute(reference);
            self.arena.as_const(reference)
        } else {
            None
        };
        match inst.mnemonic() {
            M::Mov => {
                let value = self.context_read_operand(inst, 1, address);
                self.context_write_operand(inst, 0, address, value);
            }
            M::Xchg => {
                if memory {
                    self.context_tracker.invalidate();
                    return;
                }
                let left = self.context_read_operand(inst, 0, address);
                let right = self.context_read_operand(inst, 1, address);
                self.context_write_operand(inst, 0, address, right);
                self.context_write_operand(inst, 1, address, left);
            }
            M::Not => {
                let value = self.context_read_operand(inst, 0, address).not();
                self.context_write_operand(inst, 0, address, value);
            }
            M::Lea if inst.memory_index() == Register::None => {
                let value = super::decode_reg(inst.memory_base())
                    .filter(|(_, offset, width)| *offset == 0 && *width == super::Width::W64)
                    .map(|(reg, _, _)| {
                        self.context_tracker
                            .reg(reg)
                            .offset(inst.memory_displacement64())
                    })
                    .unwrap_or_default();
                self.context_write_operand(inst, 0, None, value);
            }
            M::Add | M::Sub
                if matches!(
                    inst.op_kind(1),
                    OpKind::Immediate8to64 | OpKind::Immediate32to64 | OpKind::Immediate64
                ) =>
            {
                let n = match inst.op_kind(1) {
                    OpKind::Immediate8to64 => inst.immediate8to64() as u64,
                    OpKind::Immediate32to64 => inst.immediate32to64() as u64,
                    _ => inst.immediate64(),
                };
                let n = if inst.mnemonic() == M::Sub {
                    n.wrapping_neg()
                } else {
                    n
                };
                let value = self.context_read_operand(inst, 0, address).offset(n);
                self.context_write_operand(inst, 0, address, value);
            }
            M::Push | M::Pushfq => {
                let value = if inst.mnemonic() == M::Push {
                    self.context_read_operand(inst, 0, address)
                } else {
                    Origin::default()
                };
                if let Some(sp) = self.state.concrete_rsp(&self.arena) {
                    self.context_tracker
                        .store(sp.wrapping_sub(8), 8, value, true);
                } else {
                    self.context_tracker.invalidate();
                }
                let sp = self
                    .context_tracker
                    .reg(Reg::Rsp)
                    .offset(0u64.wrapping_sub(8));
                self.context_tracker.set_reg(Reg::Rsp, sp);
            }
            M::Pop | M::Popfq => {
                let value = self
                    .state
                    .concrete_rsp(&self.arena)
                    .map(|sp| self.context_tracker.load(sp))
                    .unwrap_or_default();
                let sp = self.context_tracker.reg(Reg::Rsp).offset(8);
                self.context_tracker.set_reg(Reg::Rsp, sp);
                if inst.mnemonic() == M::Pop {
                    let address = if inst.memory_base() == Register::RSP {
                        address.map(|a| a.wrapping_add(8))
                    } else {
                        address
                    };
                    self.context_write_operand(inst, 0, address, value);
                }
            }
            // CALL provenance is consumed immediately before the modeled guest
            // call. An internal call invalidates its implicit stack write below.
            M::Call | M::Nop | M::Pause | M::Jmp | M::Int3 => {}
            _ => {
                let mut factory = InstructionInfoFactory::new();
                let info = factory.info(inst);
                for used in info.used_registers() {
                    if matches!(
                        used.access(),
                        OpAccess::Write
                            | OpAccess::CondWrite
                            | OpAccess::ReadWrite
                            | OpAccess::ReadCondWrite
                    ) {
                        if let Some((r, _, _)) = super::decode_reg(used.register()) {
                            self.context_tracker.set_reg(r, Origin::default());
                        }
                    }
                }
                for used in info.used_memory() {
                    if matches!(
                        used.access(),
                        OpAccess::Write
                            | OpAccess::CondWrite
                            | OpAccess::ReadWrite
                            | OpAccess::ReadCondWrite
                    ) {
                        if let Some(a) = address {
                            self.context_tracker.store(
                                a,
                                used.memory_size().size() as u32,
                                Origin::default(),
                                false,
                            );
                        } else {
                            self.context_tracker.invalidate();
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Image {
    base: u64,
    layout_index: usize,
}
impl Image {
    fn layout(self) -> Layout {
        LAYOUTS[self.layout_index]
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct ContextTracker {
    regs: HashMap<Reg, Origin>,
    memory: BTreeMap<u64, Origin>,
    epoch: u64,
    pending: Option<Image>,
    current: Option<Image>,
    /// Regions which received an executed fresh whole-register save since the
    /// last unknown effect. Known subsequent stores can change their contents;
    /// each location's present value is separately checked before seeding. This
    /// records storage provenance, never which image the next consumer uses.
    saved_images: Vec<Image>,
    /// Once an actual restore was seen, content scoring cannot resurrect a
    /// historical mirror when the current round trip is incomplete.
    pub required: bool,
}

impl ContextTracker {
    pub fn reg(&self, r: Reg) -> Origin {
        self.regs.get(&r).copied().unwrap_or_default()
    }
    pub fn set_reg(&mut self, r: Reg, value: Origin) {
        self.regs.insert(r, value);
    }
    pub fn current(&self) -> Option<(u64, usize)> {
        self.current.map(|i| (i.base, i.layout_index))
    }
    pub fn saved_images(&self) -> Vec<(u64, usize)> {
        self.saved_images
            .iter()
            .map(|image| (image.base, image.layout_index))
            .collect()
    }
    pub fn invalidate(&mut self) {
        self.current = None;
        self.pending = None;
        self.memory.clear();
        self.regs.clear();
        self.saved_images.clear();
    }
    pub fn load(&self, address: u64) -> Origin {
        let mut value = self.memory.get(&address).copied().unwrap_or_default();
        // Only PUSH temporaries retain a prior read location. Regular stores
        // keep return-save tags, while a fresh load names its actual source.
        if value.read.is_none() {
            value.read = Some(Transform::new(address));
        }
        // A historical image location cannot recreate a call-result tag after
        // its contents changed. Only actual register-save/copy tags may flow.
        value
    }
    pub fn store(&mut self, address: u64, bytes: u32, mut value: Origin, push_temporary: bool) {
        if bytes == 0 {
            self.invalidate();
            return;
        }
        let lo = address.saturating_sub(7);
        let Some(hi) = address.checked_add(bytes as u64) else {
            self.invalidate();
            return;
        };
        let overlaps: Vec<_> = self.memory.range(lo..hi).map(|(&a, _)| a).collect();
        for a in overlaps {
            self.memory.remove(&a);
        }
        // A load followed by an overwrite no longer proves that its register
        // equals the current image slot at the later CALL boundary.
        for old in self.regs.values_mut().chain(self.memory.values_mut()) {
            if old
                .read
                .is_some_and(|r| r.root < hi && r.root.saturating_add(8) > address)
            {
                old.read = None;
            }
        }
        if value
            .read
            .is_some_and(|r| r.root < hi && r.root.saturating_add(8) > address)
        {
            value.read = None;
        }
        if bytes != 8 {
            return;
        }
        if !push_temporary {
            value.read = None;
        }
        if value.read.is_some() || value.saved.is_some() {
            self.memory.insert(address, value);
        }
        if let Some((reg, epoch)) = value.saved.and_then(Transform::exact) {
            if epoch == self.epoch {
                for (layout_index, layout) in LAYOUTS.iter().enumerate() {
                    let Some(offset) = layout.slot(0, reg) else {
                        continue;
                    };
                    let Some(base) = address.checked_sub(offset) else {
                        continue;
                    };
                    let image = Image { base, layout_index };
                    if base.checked_add(layout.first_reg + 128).is_some()
                        && self.complete_save(image)
                        && !self.saved_images.contains(&image)
                    {
                        self.saved_images.push(image);
                    }
                }
            }
        }
        if let Some(pending) = self.pending {
            if self.complete_save(pending) {
                self.current = Some(pending);
                self.pending = None;
            }
        } else if let Some(current) = self.current {
            // A complete copy to a different image loses the unique active
            // location. A subsequent executed round trip must establish it.
            if let Some((reg, epoch)) = value.saved.and_then(Transform::exact) {
                if epoch == self.epoch {
                    for (layout_index, layout) in LAYOUTS.iter().enumerate() {
                        let Some(offset) = layout.slot(0, reg) else {
                            continue;
                        };
                        let Some(base) = address.checked_sub(offset) else {
                            continue;
                        };
                        let image = Image { base, layout_index };
                        if image != current && self.complete_save(image) {
                            self.current = None;
                            break;
                        }
                    }
                }
            }
        }
    }
    fn complete_save(&self, image: Image) -> bool {
        image.layout().order.iter().all(|&reg| {
            image
                .layout()
                .slot(image.base, reg)
                .and_then(|a| self.memory.get(&a))
                .and_then(|v| v.saved)
                .and_then(Transform::exact)
                == Some((reg, self.epoch))
        })
    }
    fn restored_image(&self) -> Option<Image> {
        let mut found = None;
        for (layout_index, layout) in LAYOUTS.iter().enumerate() {
            let first = layout.order[0];
            let Some(address) = self.reg(first).read.and_then(Transform::exact) else {
                continue;
            };
            let Some(base) = address.checked_sub(layout.first_reg) else {
                continue;
            };
            if layout
                .order
                .iter()
                .all(|&r| self.reg(r).read.and_then(Transform::exact) == layout.slot(base, r))
            {
                if found.is_some() {
                    return None;
                }
                found = Some(Image { base, layout_index });
            }
        }
        found
    }
    pub fn guest_call(&mut self, in_vm: bool) {
        let restored = in_vm.then(|| self.restored_image()).flatten();
        self.required |= restored.is_some();
        self.current = None;
        self.pending = restored;
        self.saved_images.clear();
        self.epoch = self.epoch.wrapping_add(1);
        self.memory.clear();
        self.regs.clear();
        for r in super::super::expr::GPRS {
            self.regs.insert(
                r,
                Origin {
                    read: None,
                    saved: Some(Transform::new((r, self.epoch))),
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: u64 = 0x7fff_0000_1000;

    fn restored(layout_index: usize) -> ContextTracker {
        let mut tracker = ContextTracker::default();
        for &reg in LAYOUTS[layout_index].order {
            let address = LAYOUTS[layout_index].slot(BASE, reg).unwrap();
            tracker.set_reg(reg, tracker.load(address));
        }
        tracker
    }

    fn save(tracker: &mut ContextTracker, base: u64, layout_index: usize, omit: Option<Reg>) {
        for &reg in LAYOUTS[layout_index].order {
            if Some(reg) != omit {
                tracker.store(
                    LAYOUTS[layout_index].slot(base, reg).unwrap(),
                    8,
                    tracker.reg(reg),
                    false,
                );
            }
        }
    }

    #[test]
    fn context_origin_requires_every_restore_and_return_save_slot() {
        for &missing in LAYOUTS[0].order {
            let mut missing_restore = restored(0);
            missing_restore.set_reg(missing, Origin::default());
            missing_restore.guest_call(true);
            save(&mut missing_restore, BASE, 0, None);
            assert!(!missing_restore.required);
            assert!(missing_restore.current().is_none());

            let mut missing_save = restored(0);
            missing_save.guest_call(true);
            save(&mut missing_save, BASE, 0, Some(missing));
            assert!(missing_save.required);
            assert!(missing_save.current().is_none());
        }
    }

    #[test]
    fn context_origin_rejects_permuted_or_unresolved_restore_transforms() {
        for reg in [Reg::Rax, Reg::Rsp, Reg::R15] {
            let mut tracker = restored(0);
            tracker.set_reg(reg, tracker.reg(reg).not());
            tracker.guest_call(true);
            assert!(!tracker.required);
            assert!(tracker.current().is_none());
        }
        let mut tracker = restored(0);
        tracker.set_reg(Reg::Rdi, tracker.reg(Reg::Rsi));
        tracker.guest_call(true);
        assert!(!tracker.required);
    }

    #[test]
    fn context_origin_overwritten_restore_sources_do_not_survive_until_call() {
        for bytes in [1, 2, 4, 8] {
            let mut tracker = restored(0);
            let slot = LAYOUTS[0].slot(BASE, Reg::Rdi).unwrap();
            tracker.store(slot + 1, bytes, Origin::default(), false);
            tracker.guest_call(true);
            assert!(!tracker.required, "an overwritten load is historical");
        }
    }

    #[test]
    fn context_origin_new_call_and_ambiguous_complete_copy_expire_the_candidate() {
        let mut tracker = restored(0);
        tracker.guest_call(true);
        save(&mut tracker, BASE, 0, None);
        assert_eq!(tracker.current(), Some((BASE, 0)));
        tracker.guest_call(true);
        assert!(tracker.required);
        assert!(tracker.current().is_none());
        save(&mut tracker, BASE, 0, None);
        assert!(
            tracker.current().is_none(),
            "fresh saves need a fresh complete restore"
        );

        let mut tracker = restored(0);
        tracker.guest_call(true);
        save(&mut tracker, BASE, 0, None);
        for &reg in LAYOUTS[0].order {
            let value = tracker.load(LAYOUTS[0].slot(BASE, reg).unwrap());
            tracker.store(LAYOUTS[1].slot(BASE + 0x200, reg).unwrap(), 8, value, false);
        }
        assert!(
            tracker.current().is_none(),
            "two complete images need an actual consumer proof"
        );
    }

    #[test]
    fn context_origin_unknown_write_or_non_vm_call_expires_the_candidate() {
        for (address, width) in [(BASE, 0), (u64::MAX - 3, 8)] {
            let mut tracker = restored(0);
            tracker.guest_call(true);
            save(&mut tracker, BASE, 0, None);
            tracker.store(address, width, Origin::default(), false);
            assert!(tracker.required);
            assert!(tracker.current().is_none());
            assert!(tracker.saved_images().is_empty());
        }
        let mut tracker = restored(0);
        tracker.guest_call(false);
        save(&mut tracker, BASE, 0, None);
        assert!(!tracker.required);
        assert!(tracker.current().is_none());
    }

    #[test]
    fn location_modified_image_copy_cannot_recreate_fresh_register_tags() {
        for bytes in [1, 8] {
            let mut tracker = restored(0);
            tracker.guest_call(true);
            save(&mut tracker, BASE, 0, None);
            assert!(tracker.saved_images().contains(&(BASE, 0)));
            let slot = LAYOUTS[0].slot(BASE, Reg::Rdi).unwrap();
            tracker.store(slot, bytes, Origin::default(), false);
            // The old region is still qualified storage. Its overwritten slot
            // is not a fresh return-save value and cannot qualify a new image.
            assert!(tracker.saved_images().contains(&(BASE, 0)));
            assert!(tracker.load(slot).saved.is_none());
            let copied = BASE + 0x400;
            for &reg in LAYOUTS[0].order {
                let value = tracker.load(LAYOUTS[0].slot(BASE, reg).unwrap());
                tracker.store(LAYOUTS[1].slot(copied, reg).unwrap(), 8, value, false);
            }
            assert!(!tracker.saved_images().contains(&(copied, 1)));
        }
    }
}
