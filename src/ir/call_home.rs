//! Function-wide homes for exact, non-reexecuted call results whose definition
//! dominates every use. This never establishes a native/VM join or memory phi.

use crate::ir::expr::{Arena, CallOrigin, Op, Ref};
use crate::ir::lift::Event;
use crate::ir::{Block, BlockId, Cfg, Terminator};
use std::collections::{HashMap, HashSet};

#[cfg(test)]
#[path = "call_home_tests.rs"]
mod tests;

#[derive(Default)]
pub(crate) struct CallHomePlan {
    slots: HashMap<CallOrigin, u32>,
}

#[derive(Clone, Copy)]
struct Use {
    block: BlockId,
    /// Event operands are read before that event's result is defined.
    position: usize,
}

fn successors(block: &Block) -> Vec<BlockId> {
    let mut out = match &block.terminator {
        Terminator::Jump(target) | Terminator::Backedge { target } => vec![*target],
        Terminator::Branch { taken, not_taken, .. }
        | Terminator::Adopted { taken, not_taken } => vec![*taken, *not_taken],
        Terminator::Switch { targets, .. }
        | Terminator::AdoptedSwitch { targets } => targets.clone(),
        _ => Vec::new(),
    };
    out.sort_unstable();
    out.dedup();
    out
}

fn event_inputs(event: &Event) -> Vec<Ref> {
    match event {
        Event::StackProbe { top, size, .. } => vec![*top, *size],
        Event::Store { addr, value, .. } | Event::Load { addr, value, .. } => {
            vec![*addr, *value]
        }
        Event::Call { target, args, .. } => target.iter().copied()
            .chain(args.iter().map(|(_, value)| *value)).collect(),
        Event::Boxed { mem, uses, .. } => mem.iter().map(|m| m.addr)
            .chain(uses.iter().map(|(_, value)| *value)).collect(),
    }
}

fn record_uses(arena: &Arena, roots: Vec<Ref>, at: Use,
    uses: &mut HashMap<CallOrigin, Vec<Use>>) {
    let mut pending = roots;
    let mut seen = HashSet::new();
    let mut origins = HashSet::new();
    while let Some(value) = pending.pop() {
        if !seen.insert(value) { continue; }
        if matches!(arena.op(value), Op::Opaque("call_ret", _)) {
            if let Some(origin) = arena.call_origin(value) { origins.insert(origin); }
        }
        pending.extend(arena.children(value));
    }
    for origin in origins { uses.entry(origin).or_default().push(at); }
}

/// An origin in a cyclic definition block may denote different live dynamic
/// instances. A single last-result slot is deliberately not a loop value phi.
fn definition_is_cyclic(owner: BlockId, edges: &HashMap<BlockId, Vec<BlockId>>) -> bool {
    let mut pending = edges[&owner].clone();
    let mut seen = HashSet::new();
    while let Some(block) = pending.pop() {
        if block == owner { return true; }
        if seen.insert(block) { pending.extend(edges[&block].iter().copied()); }
    }
    false
}

/// Removing the definition must disconnect every use from the entry. A diamond
/// or a loop containing only uses does not create another dynamic call result:
/// `definition_is_cyclic` separately proves that the defining call cannot run
/// again. Include unreachable blocks as conservative unknown roots, so a dead
/// predecessor cannot accidentally supply a proof for a reachable use.
fn reachable_without_definition(owner: BlockId, entry: BlockId,
    edges: &HashMap<BlockId, Vec<BlockId>>, reachable: &HashSet<BlockId>) -> HashSet<BlockId> {
    let mut pending = vec![entry];
    pending.extend(edges.keys().copied().filter(|id| !reachable.contains(id)));
    let mut seen = HashSet::new();
    while let Some(block) = pending.pop() {
        if block == owner || !seen.insert(block) { continue; }
        pending.extend(edges[&block].iter().copied());
    }
    seen
}

impl CallHomePlan {
    pub(crate) fn build(cfg: &Cfg) -> Self {
        let mut result = Self::default();
        if cfg.timed_out || cfg.unresolved != 0 { return result; }
        let blocks: HashMap<_, _> = cfg.blocks.iter().map(|b| (b.id,b)).collect();
        if blocks.len() != cfg.blocks.len() || !blocks.contains_key(&cfg.entry) { return result; }
        let edges: HashMap<_, _> = cfg.blocks.iter().map(|b| (b.id,successors(b))).collect();
        if edges.values().flatten().any(|id| !blocks.contains_key(id)) { return result; }
        let mut reachable = HashSet::new();
        let mut pending = vec![cfg.entry];
        while let Some(block) = pending.pop() {
            if reachable.insert(block) { pending.extend(edges[&block].iter().copied()); }
        }
        let mut definitions: HashMap<CallOrigin, Vec<(BlockId,usize,u64)>> = HashMap::new();
        let mut uses = HashMap::new();
        for block in &cfg.blocks {
            for (position,event) in block.events.iter().enumerate() {
                if let Event::Call { ret: Some(value), site, .. } = event {
                    if let Some(origin) = cfg.arena.call_origin(*value) {
                        definitions.entry(origin).or_default().push((block.id,position,*site));
                    }
                }
                record_uses(&cfg.arena,event_inputs(event),Use {block:block.id,position},&mut uses);
            }
            let mut roots: Vec<_> = block.exit_regs.iter().map(|(_,value)| *value).collect();
            match block.terminator {
                Terminator::Branch { predicate, .. } => roots.push(predicate),
                Terminator::Switch { selector, .. } => roots.push(selector),
                Terminator::Return { dest } => roots.push(dest),
                Terminator::IndirectTailCall { target } => roots.push(target),
                _ => {},
            }
            record_uses(&cfg.arena,roots,Use {block:block.id,position:block.events.len()},&mut uses);
        }
        let mut accepted = Vec::new();
        for (origin, locations) in uses {
            if !reachable.contains(&origin.block)
                || !locations.iter().any(|use_| use_.block != origin.block) { continue; }
            let Some(defs) = definitions.get(&origin) else { continue; };
            if defs.as_slice() != [(origin.block,origin.event_index,origin.site)] { continue; }
            if definition_is_cyclic(origin.block,&edges) { continue; }
            let bypassed = reachable_without_definition(origin.block,cfg.entry,&edges,&reachable);
            if locations.iter().all(|use_| {
                reachable.contains(&use_.block) && if use_.block == origin.block {
                    use_.position > origin.event_index
                } else {
                    !bypassed.contains(&use_.block)
                }
            }) { accepted.push(origin); }
        }
        accepted.sort_by_key(|origin| (origin.block,origin.event_index,origin.site));
        for (slot,origin) in accepted.into_iter().enumerate() {
            let Ok(slot) = u32::try_from(slot) else { return Self::default(); };
            result.slots.insert(origin,slot);
        }
        result
    }

    pub(crate) fn slot_for(&self, origin: CallOrigin) -> Option<u32> {
        self.slots.get(&origin).copied()
    }

    #[cfg(test)]
    fn len(&self) -> usize { self.slots.len() }
}
