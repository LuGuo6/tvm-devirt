//! Conservative experimental join proof. Only bytes actually parameterized by
//! the plan receive Home markers. Everything else must be an identical pure
//! entry expression; old dynamic identities are deliberately not guessed.
use super::*;
use crate::ir::expr::{self, BinOp, BlockRef, UnOp};
use crate::vm::state::Byte;
use std::collections::BTreeMap;

type ProofResult<T> = Result<T, String>;
const CALL_PROVENANCE_NODE_BUDGET: usize = 100_000;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum PureOp {
    Const(u64),
    Init(Reg),
    Bin(BinOp, usize, usize),
    Un(UnOp, usize),
    Zext(usize),
    Sext(usize),
    Trunc(usize),
    Select(usize, usize, usize),
}

#[derive(Default)]
struct PureInterner {
    nodes: HashMap<(Width, PureOp), usize>,
}

impl PureInterner {
    fn key(
        &mut self,
        arena: &Arena,
        value: Ref,
        memo: &mut HashMap<Ref, usize>,
        depth: usize,
    ) -> ProofResult<usize> {
        if let Some(&key) = memo.get(&value) {
            return Ok(key);
        }
        if depth > 128 || self.nodes.len() > 100_000 {
            return Err("entry-state pure-expression proof budget exhausted".into());
        }
        let next = depth + 1;
        let op = match *arena.op(value) {
            Op::Const(c) => PureOp::Const(c),
            Op::InitReg(reg) => PureOp::Init(reg),
            Op::Bin(op, a, b) => PureOp::Bin(
                op,
                self.key(arena, a, memo, next)?,
                self.key(arena, b, memo, next)?,
            ),
            Op::Un(op, a) => PureOp::Un(op, self.key(arena, a, memo, next)?),
            Op::Zext(a) => PureOp::Zext(self.key(arena, a, memo, next)?),
            Op::Sext(a) => PureOp::Sext(self.key(arena, a, memo, next)?),
            Op::Trunc(a) => PureOp::Trunc(self.key(arena, a, memo, next)?),
            Op::Select(c, a, b) => PureOp::Select(
                self.key(arena, c, memo, next)?,
                self.key(arena, a, memo, next)?,
                self.key(arena, b, memo, next)?,
            ),
            Op::Load(..) | Op::Opaque(..) | Op::Param(..) => {
                return Err(format!(
                    "unseeded dynamic entry value {} has no cross-edge identity",
                    expr::render(arena, value, 2)
                ));
            }
        };
        let count = self.nodes.len();
        let key = *self.nodes.entry((arena.width(value), op)).or_insert(count);
        memo.insert(value, key);
        Ok(key)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ByteKey {
    Const(u8),
    Pure(usize, u32),
    Home(Reg, u32),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct EntrySignature {
    memory: BTreeMap<u64, ByteKey>,
    regs: BTreeMap<u8, usize>,
    flags: BTreeMap<u8, usize>,
    xmm: BTreeMap<(u8, u8), usize>,
    pins: Vec<(usize, u64)>,
    value_sets: Vec<(usize, u64)>,
    written_args: Vec<u8>,
    // Each group names all seeded homes holding one live call-origin value.
    // Arena-local Ref/Opaque IDs are never compared across incoming edges.
    call_result_homes: Vec<Vec<u8>>,
    call_output_slots: Vec<u64>,
    vip_slot: Option<u64>,
    unwind_slot: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
struct Location {
    address: u64,
    home: Reg,
    image: (u64, usize),
    seed: bool,
}

pub(crate) struct LocationHomePlan {
    canonical: (u64, usize),
    rsp: u64,
    locations: Vec<Location>,
    interner: PureInterner,
    signature: Option<EntrySignature>,
}

impl LocationHomePlan {
    fn live_call_results(
        emu: &Emulator<'_>,
        value_sets: &[(Ref, u64)],
        node_budget: usize,
    ) -> ProofResult<HashSet<Ref>> {
        let mut pending = emu.state.symbolic_roots();
        pending.extend(emu.pins.iter().map(|(root, _)| *root));
        pending.extend(value_sets.iter().map(|(root, _)| *root));
        // Events describe effects already executed. Their historical call
        // returns cannot make an otherwise dead value part of entry state.
        Self::call_results_from_roots(emu, pending, node_budget)
    }

    fn call_results_from_roots(
        emu: &Emulator<'_>,
        mut pending: Vec<Ref>,
        node_budget: usize,
    ) -> ProofResult<HashSet<Ref>> {
        let mut visited = HashSet::new();
        let mut live = HashSet::new();
        while let Some(value) = pending.pop() {
            if !visited.insert(value) {
                continue;
            }
            if visited.len() > node_budget {
                return Err("entry-state call-provenance proof budget exhausted".into());
            }
            if emu.guest_call_results.contains(&value) {
                live.insert(value);
            }
            pending.extend(emu.arena.children(value));
        }
        Ok(live)
    }

    fn call_result_homes(
        &self,
        emu: &Emulator<'_>,
        value_sets: &[(Ref, u64)],
    ) -> ProofResult<Vec<Vec<u8>>> {
        let mut by_value: HashMap<Ref, Vec<u8>> = HashMap::new();
        for location in self.locations.iter().filter(|location| location.seed) {
            if let Some(value) = Self::natural(emu, location.address) {
                by_value.entry(value).or_default().push(location.home as u8);
            }
        }
        let live = Self::live_call_results(emu, value_sets, CALL_PROVENANCE_NODE_BUDGET)?;
        if !live.is_empty() {
            for &value in by_value.keys() {
                let sources = Self::call_results_from_roots(
                    emu, vec![value], CALL_PROVENANCE_NODE_BUDGET,
                )?;
                if sources.iter().any(|&source| source != value) {
                    return Err("live call-result provenance has no seeded home for a nested use".into());
                }
            }
            let seeded_addresses: HashSet<_> = self.locations.iter()
                .filter(|location| location.seed)
                .flat_map(|location| location.address..location.address + 8)
                .collect();
            let mut unchanged_roots: Vec<_> = emu.state.regs.values().copied()
                .chain(emu.state.flags.values().copied())
                .chain(emu.state.xmm.values().copied())
                .chain(emu.pins.iter().map(|(root, _)| *root))
                .chain(value_sets.iter().map(|(root, _)| *root))
                .collect();
            unchanged_roots.extend(emu.state.mem.iter().filter_map(|(address, byte)| {
                match byte {
                    Byte::Sym { expr, .. } if !seeded_addresses.contains(address) => Some(*expr),
                    _ => None,
                }
            }));
            // The signature already rejects symbolic stores/forwarding. This
            // leaves every place where seed() will retain the old expression.
            if !Self::call_results_from_roots(emu, unchanged_roots, CALL_PROVENANCE_NODE_BUDGET)?.is_empty() {
                return Err("live call-result provenance remains in unseeded state".into());
            }
        }
        let mut normalized = Vec::new();
        for value in live {
            // A source nested within a home expression is not that home.
            // Dropping it or attributing its provenance to the whole expression
            // would both change classification after the join.
            let mut homes = by_value.remove(&value)
                .ok_or("live call-result provenance has no seeded home")?;
            homes.sort_unstable();
            homes.dedup();
            normalized.push(homes);
        }
        normalized.sort_unstable();
        Ok(normalized)
    }

    fn frame(emu: &Emulator<'_>, homes: &[(Reg, Ref)]) -> ProofResult<((u64, usize), u64)> {
        if !emu.diagnostic_location_homes || emu.diagnostic_context_origin {
            return Err("location-home proof requires its exclusive diagnostic mode".into());
        }
        let base = emu.guest_ctx.ok_or("missing parent context descriptor")?;
        let layout = emu.guest_layout.ok_or("missing parent context layout")?;
        let index = LAYOUTS
            .iter()
            .position(|candidate| *candidate == layout)
            .ok_or("unsupported context layout")?;
        let mut seen = HashSet::new();
        if homes.len() != 16 || homes.iter().any(|(r, _)| !seen.insert(*r)) {
            return Err("parent exit snapshot must supply every GPR exactly once".into());
        }
        let rsp = homes
            .iter()
            .find(|(r, _)| *r == Reg::Rsp)
            .and_then(|(_, value)| emu.arena.as_const(*value))
            .ok_or("parent guest RSP is not concrete")?;
        if rsp.abs_diff(emu.stack_base) > crate::ir::sched::FRAME_WINDOW {
            return Err("parent guest RSP lies outside the frame window".into());
        }
        Ok(((base, index), rsp))
    }

    fn qualified(emu: &Emulator<'_>, image: (u64, usize), address: u64, rsp: u64) -> bool {
        let Some(end) = address.checked_add(8) else {
            return false;
        };
        let layout = LAYOUTS[image.1];
        let Some(start) = image.0.checked_add(layout.first_reg) else {
            return false;
        };
        let Some(image_end) = start.checked_add(128) else {
            return false;
        };
        emu.context_tracker.saved_images().contains(&image)
            && address >= start
            && end <= image_end
            && address % 8 == 0
            && address
                >= emu
                    .stack_base
                    .saturating_sub(crate::ir::sched::FRAME_WINDOW)
            && end
                <= emu
                    .stack_base
                    .saturating_add(crate::ir::sched::FRAME_WINDOW)
            && end <= rsp
            && (address..end).all(|a| emu.pe.section_for_va(a).is_none())
    }

    fn natural(emu: &Emulator<'_>, address: u64) -> Option<Ref> {
        address.checked_add(8)?;
        let Byte::Sym { expr, index: 0 } = *emu.state.mem.get(&address)? else {
            return None;
        };
        if emu.arena.width(expr) != Width::W64 {
            return None;
        }
        (0..8)
            .all(|i| {
                emu.state.mem.get(&(address + i))
                    == Some(&Byte::Sym {
                        expr,
                        index: i as u32,
                    })
            })
            .then_some(expr)
    }

    fn qword(emu: &mut Emulator<'_>, address: u64) -> ProofResult<Ref> {
        if address.checked_add(8).is_none() {
            return Err("qword address overflows".into());
        }
        if let Some(value) = Self::natural(emu, address) {
            return Ok(value);
        }
        let mut value = 0;
        for i in 0..8 {
            let Some(Byte::Const(byte)) = emu.state.mem.get(&(address + i)) else {
                return Err(format!(
                    "qword {address:#x} is missing, partial or not a natural W64 value"
                ));
            };
            value |= (*byte as u64) << (i * 8);
        }
        Ok(emu.arena.constant(value, Width::W64))
    }

    fn add(&mut self, location: Location) -> ProofResult<()> {
        for prior in &self.locations {
            if prior.address == location.address {
                return if prior.home == location.home && prior.seed == location.seed {
                    Ok(())
                } else {
                    Err("the same qword has conflicting register homes".into())
                };
            }
            if prior.address < location.address + 8 && location.address < prior.address + 8 {
                return Err("overlapping qwords in the location-home plan".into());
            }
        }
        self.locations.push(location);
        Ok(())
    }

    pub(crate) fn discover(
        emu: &mut Emulator<'_>,
        homes: &[(Reg, Ref)],
        value_sets: &[(Ref, u64)],
    ) -> ProofResult<Self> {
        let (canonical, rsp) = Self::frame(emu, homes)?;
        let mut plan = Self {
            canonical,
            rsp,
            locations: Vec::new(),
            interner: PureInterner::default(),
            signature: None,
        };
        // Every canonical slot participates, including constants and the
        // concrete RSP anchor. There is no separate legacy seeding path.
        for &(home, _) in homes {
            let address = LAYOUTS[canonical.1]
                .slot(canonical.0, home)
                .ok_or("canonical slot missing")?;
            if !Self::qualified(emu, canonical, address, rsp) {
                return Err(format!(
                    "canonical qword {address:#x} lacks executed-save/private-frame provenance"
                ));
            }
            plan.add(Location {
                address,
                home,
                image: canonical,
                seed: home != Reg::Rsp,
            })?;
        }
        let mut images = emu.context_tracker.saved_images();
        images.sort_unstable();
        let mut ordered = homes.to_vec();
        ordered.sort_by_key(|(reg, _)| *reg as u8);
        for image in images {
            for &role in LAYOUTS[image.1].order {
                let address = LAYOUTS[image.1]
                    .slot(image.0, role)
                    .ok_or("mirror slot missing")?;
                if plan
                    .locations
                    .iter()
                    .any(|location| location.address == address)
                {
                    continue;
                }
                if !Self::qualified(emu, image, address, rsp) {
                    continue;
                }
                let Some(value) = Self::natural(emu, address) else {
                    continue;
                };
                if !expr::leaves(&emu.arena, value).iter().any(|leaf| {
                    matches!(
                        emu.arena.op(*leaf),
                        Op::Load(..) | Op::Opaque(..) | Op::Param(..)
                    )
                }) {
                    continue;
                }
                let matching = ordered
                    .iter()
                    .filter(|(reg, v)| *reg != Reg::Rsp && *v == value);
                let chosen = matching
                    .clone()
                    .find(|(reg, _)| *reg == role)
                    .or_else(|| matching.into_iter().next());
                if let Some(&(home, _)) = chosen {
                    plan.add(Location {
                        address,
                        home,
                        image,
                        seed: true,
                    })?;
                }
            }
        }
        plan.validate(emu, homes, value_sets)?;
        Ok(plan)
    }

    fn signature(
        &mut self,
        emu: &Emulator<'_>,
        value_sets: &[(Ref, u64)],
    ) -> ProofResult<EntrySignature> {
        if !emu.state.sym_stores.is_empty() || emu.state.has_symbolic_forwarding() {
            return Err(
                "symbolic-memory history or forwarding is not proven equal across incoming edges"
                    .into(),
            );
        }
        let mut seeded = HashMap::new();
        for location in self.locations.iter().filter(|location| location.seed) {
            for i in 0..8 {
                seeded.insert(location.address + i, (location.home, i as u32));
            }
        }
        let mut memo = HashMap::new();
        let mut memory = BTreeMap::new();
        let mut addresses: Vec<_> = emu.state.mem.keys().copied().collect();
        addresses.sort_unstable();
        for address in addresses {
            let byte = if let Some(&(home, index)) = seeded.get(&address) {
                ByteKey::Home(home, index)
            } else {
                match emu.state.mem[&address] {
                    Byte::Const(byte) => ByteKey::Const(byte),
                    Byte::Sym { expr, index } => ByteKey::Pure(
                        self.interner
                            .key(&emu.arena, expr, &mut memo, 0)
                            .map_err(|why| format!("unseeded memory byte {address:#x}: {why}"))?,
                        index,
                    ),
                }
            };
            memory.insert(address, byte);
        }
        let mut regs = BTreeMap::new();
        let mut ordered_regs: Vec<_> = emu
            .state
            .regs
            .iter()
            .map(|(&reg, &value)| (reg, value))
            .collect();
        ordered_regs.sort_by_key(|(reg, _)| *reg as u8);
        for (reg, value) in ordered_regs {
            regs.insert(
                reg as u8,
                self.interner
                    .key(&emu.arena, value, &mut memo, 0)
                    .map_err(|why| format!("physical {}: {why}", reg.name()))?,
            );
        }
        let mut flags = BTreeMap::new();
        let mut ordered_flags: Vec<_> = emu
            .state
            .flags
            .iter()
            .map(|(&flag, &value)| (flag, value))
            .collect();
        ordered_flags.sort_by_key(|(flag, _)| *flag as u8);
        for (flag, value) in ordered_flags {
            flags.insert(
                flag as u8,
                self.interner
                    .key(&emu.arena, value, &mut memo, 0)
                    .map_err(|why| format!("flag {flag:?}: {why}"))?,
            );
        }
        let mut xmm = BTreeMap::new();
        let mut ordered_xmm: Vec<_> = emu
            .state
            .xmm
            .iter()
            .map(|(&(reg, half), &value)| ((reg, half as u8), value))
            .collect();
        ordered_xmm.sort_by_key(|(reg, _)| *reg);
        for ((reg, half), value) in ordered_xmm {
            xmm.insert(
                (reg, half),
                self.interner
                    .key(&emu.arena, value, &mut memo, 0)
                    .map_err(|why| format!("XMM{reg} half {half}: {why}"))?,
            );
        }
        let pins = emu
            .pins
            .iter()
            .enumerate()
            .map(|(index, &(value, pin))| {
                Ok((
                    self.interner
                        .key(&emu.arena, value, &mut memo, 0)
                        .map_err(|why| format!("pin {index}: {why}"))?,
                    pin,
                ))
            })
            .collect::<ProofResult<_>>()?;
        let normalized_value_sets = value_sets
            .iter()
            .enumerate()
            .map(|(index, &(value, set))| {
                Ok((
                    self.interner
                        .key(&emu.arena, value, &mut memo, 0)
                        .map_err(|why| format!("value set {index}: {why}"))?,
                    set,
                ))
            })
            .collect::<ProofResult<_>>()?;
        let mut written_args: Vec<_> = emu.written_args.iter().map(|reg| *reg as u8).collect();
        written_args.sort_unstable();
        let call_result_homes = self.call_result_homes(emu, value_sets)?;
        let mut call_output_slots: Vec<_> = emu.guest_call_output_slots.iter().copied().collect();
        call_output_slots.sort_unstable();
        Ok(EntrySignature {
            memory,
            regs,
            flags,
            xmm,
            pins,
            value_sets: normalized_value_sets,
            written_args,
            call_result_homes,
            call_output_slots,
            vip_slot: emu.vip_slot,
            unwind_slot: emu.unwind_slot,
        })
    }

    pub(crate) fn validate(
        &mut self,
        emu: &mut Emulator<'_>,
        homes: &[(Reg, Ref)],
        value_sets: &[(Ref, u64)],
    ) -> ProofResult<()> {
        let (canonical, rsp) = Self::frame(emu, homes)?;
        if canonical != self.canonical || rsp != self.rsp {
            return Err("incoming canonical image or concrete guest RSP changed".into());
        }
        for location in &self.locations {
            if !Self::qualified(emu, location.image, location.address, rsp) {
                return Err(format!(
                    "incoming qword {:#x} lost executed-save/private-frame provenance",
                    location.address
                ));
            }
            let value = Self::qword(emu, location.address)?;
            let expected = homes
                .iter()
                .find(|(reg, _)| *reg == location.home)
                .map(|(_, value)| *value)
                .ok_or("incoming home is absent")?;
            if value != expected {
                return Err(format!(
                    "incoming qword {:#x} does not equal parent exit {}",
                    location.address,
                    location.home.name()
                ));
            }
        }
        let signature = self.signature(emu, value_sets)?;
        if let Some(old) = &self.signature {
            if old.memory != signature.memory {
                let mut keys: Vec<_> = old
                    .memory
                    .keys()
                    .chain(signature.memory.keys())
                    .copied()
                    .collect();
                keys.sort_unstable();
                keys.dedup();
                let address = keys
                    .into_iter()
                    .find(|address| old.memory.get(address) != signature.memory.get(address))
                    .unwrap();
                return Err(format!(
                    "unseeded incoming memory byte {address:#x} differs or is missing"
                ));
            }
            for (name, different) in [
                ("physical GPR", old.regs != signature.regs),
                ("flags", old.flags != signature.flags),
                ("XMM", old.xmm != signature.xmm),
                ("pins", old.pins != signature.pins),
                ("value sets", old.value_sets != signature.value_sets),
                (
                    "written argument registers",
                    old.written_args != signature.written_args,
                ),
                ("VIP metadata", old.vip_slot != signature.vip_slot),
                ("unwind metadata", old.unwind_slot != signature.unwind_slot),
                ("call-result provenance", old.call_result_homes != signature.call_result_homes),
                ("call-output slots", old.call_output_slots != signature.call_output_slots),
            ] {
                if different {
                    return Err(format!("unseeded incoming {name} state differs"));
                }
            }
        }
        self.signature = Some(signature);
        Ok(())
    }

    pub(crate) fn seed(&self, emu: &mut Emulator<'_>, block: BlockRef) -> Vec<(Reg, Ref)> {
        let mut out = Vec::new();
        let signature = self.signature.as_ref().expect("seeding requires validated incoming state");
        // Validated live origins are represented by new block homes. Old refs
        // can remain in past events but must not describe future reads.
        emu.guest_call_results.clear();
        for location in self.locations.iter().filter(|location| location.seed) {
            let parameter = emu.arena.param(block, location.home);
            if signature.call_result_homes.iter().flatten()
                .any(|&home| home == location.home as u8)
            {
                emu.guest_call_results.insert(parameter);
            }
            emu.expire_call_output_slots(location.address, Width::W64.bytes());
            emu.state
                .store_concrete(&emu.arena, location.address, parameter, Width::W64);
            emu.arena.bump_mem_gen();
            emu.arena.note_concrete_store(location.address, Width::W64.bytes());
            if !out.iter().any(|(reg, _)| *reg == location.home) {
                out.push((location.home, parameter));
            }
        }
        out
    }
}

#[cfg(test)]
#[path = "location_home_tests.rs"]
mod tests;
