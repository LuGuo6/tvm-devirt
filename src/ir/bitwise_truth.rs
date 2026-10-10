//! Exact, bounded absorption for same-width bitwise expression networks.
//!
//! Each of at most four *exact Ref* atoms is assigned either zero or the width
//! mask in all 16 Boolean valuations. Each valuation is a u64, so every bit lane
//! retains its own constant bits. This proves all valuations independently at
//! every bit position, including arbitrary nonuniform constants. Opening only
//! AND/OR/XOR/NOT is essential: arithmetic, shifts and conversions are independent
//! opaque atoms, never Boolean operations or claims about related values.
//!
//! Four-atom proofs return a proper descendant or a constant. At most two
//! atoms additionally permit a fixed Boolean canonical with at most two new
//! operations. A separate bounded full-DAG closure check (including the opaque
//! atoms' children) requires a strict node-count decrease before construction.

use super::{Arena, BinOp, HashSet, Node, Op, Ref, UnOp, Width};

const MAX_ATOMS: usize = 4;
const ASSIGNMENTS: usize = 1 << MAX_ATOMS;
const MAX_NODES: usize = 32; // Includes the not-yet-interned root.
const MAX_VISITS: usize = 64; // Counts every visit, including memo hits.
const MAX_CLOSURE_NODES: usize = 256;
const MAX_CLOSURE_VISITS: usize = 1024;
type Truth = [u64; ASSIGNMENTS];

#[derive(Clone, Copy)]
enum Canonical {
    Not(Ref),
    Binary {
        op: BinOp,
        x: Ref,
        y: Ref,
        negate_x: bool,
        negate_y: bool,
        negate_result: bool,
    },
}

impl Canonical {
    fn roots(self) -> Vec<Ref> {
        match self {
            Self::Not(x) => vec![x],
            Self::Binary { x, y, .. } => vec![x, y],
        }
    }

    fn new_nodes(self) -> usize {
        match self {
            Self::Not(_) => 1,
            Self::Binary {
                negate_x,
                negate_y,
                negate_result,
                ..
            } => 1 + usize::from(negate_x || negate_y || negate_result),
        }
    }

    fn build(self, arena: &mut Arena, width: Width) -> Ref {
        // Raw interning avoids recursive simplification/rewrite cycles. Ref
        // ordering matches Arena::bin; these operands are never constants.
        let not = |arena: &mut Arena, r| {
            arena.intern(Node {
                op: Op::Un(UnOp::Not, r),
                width,
            })
        };
        match self {
            Self::Not(x) => not(arena, x),
            Self::Binary {
                op,
                x,
                y,
                negate_x,
                negate_y,
                negate_result,
            } => {
                let x = if negate_x { not(arena, x) } else { x };
                let y = if negate_y { not(arena, y) } else { y };
                let (x, y) = if x > y { (y, x) } else { (x, y) };
                let value = arena.intern(Node {
                    op: Op::Bin(op, x, y),
                    width,
                });
                if negate_result {
                    not(arena, value)
                } else {
                    value
                }
            }
        }
    }
}

// All 16 uniform Boolean functions of two atoms have a canonical with zero,
// one or two operations. Constants/positive atoms were already considered as
// descendants. Arbitrary masks can give a different function per bit lane;
// those are rejected here, not reduced to a scalar nonzero/true value.
fn canonical(atoms: &[Ref], truth: &Truth, mask: u64) -> Option<Canonical> {
    if atoms.is_empty() || atoms.len() > 2 {
        return None;
    }
    if truth.iter().any(|&v| v != 0 && v != mask) {
        return None;
    }
    let table = (0..4).fold(0u8, |table, i| table | (u8::from(truth[i] == mask) << i));
    let x = atoms[0];
    let y = *atoms.get(1).unwrap_or(&x);
    if table == 0b0101 {
        return truth
            .iter()
            .enumerate()
            .all(|(i, &v)| v == if i & 1 == 0 { mask } else { 0 })
            .then_some(Canonical::Not(x));
    }
    if atoms.len() == 1 {
        return None;
    }
    if table == 0b0011 {
        return truth
            .iter()
            .enumerate()
            .all(|(i, &v)| v == if i & 2 == 0 { mask } else { 0 })
            .then_some(Canonical::Not(y));
    }
    let (op, negate_x, negate_y, negate_result) = match table {
        0b1000 => (BinOp::And, false, false, false),
        0b1110 => (BinOp::Or, false, false, false),
        0b0110 => (BinOp::Xor, false, false, false),
        0b0100 => (BinOp::And, true, false, false),
        0b0010 => (BinOp::And, false, true, false),
        0b1101 => (BinOp::Or, true, false, false),
        0b1011 => (BinOp::Or, false, true, false),
        0b0111 => (BinOp::And, false, false, true),
        0b0001 => (BinOp::Or, false, false, true),
        0b1001 => (BinOp::Xor, false, false, true),
        _ => return None,
    };
    let candidate = Canonical::Binary {
        op,
        x,
        y,
        negate_x,
        negate_y,
        negate_result,
    };
    // Independently reconstruct every assignment rather than trusting the
    // compact table-to-recipe mapping above.
    let values: Truth = std::array::from_fn(|assignment| {
        let x = if (assignment & 1 != 0) ^ negate_x {
            mask
        } else {
            0
        };
        let y = if (assignment & 2 != 0) ^ negate_y {
            mask
        } else {
            0
        };
        let value = match op {
            BinOp::And => x & y,
            BinOp::Or => x | y,
            BinOp::Xor => x ^ y,
            _ => unreachable!(),
        };
        if negate_result {
            !value & mask
        } else {
            value
        }
    });
    (values == *truth).then_some(candidate)
}

fn closure_len(arena: &Arena, roots: &[Ref]) -> Option<usize> {
    let mut seen = HashSet::new();
    let mut pending = roots.to_vec();
    let mut visits_left = MAX_CLOSURE_VISITS;
    while let Some(r) = pending.pop() {
        visits_left = visits_left.checked_sub(1)?;
        if !seen.insert(r) {
            continue;
        }
        if seen.len() > MAX_CLOSURE_NODES {
            return None;
        }
        pending.extend(arena.children(r));
    }
    Some(seen.len())
}

struct ProvenNode {
    source: Ref,
    truth: Truth,
    reachable: u32,
}

struct Proof<'a> {
    arena: &'a Arena,
    width: Width,
    nodes: Vec<ProvenNode>,
    atoms: Vec<Ref>,
    visits_left: usize,
}

impl Proof<'_> {
    fn visit(&mut self, r: Ref) -> Option<usize> {
        self.visits_left = self.visits_left.checked_sub(1)?;
        if self.arena.width(r) != self.width {
            return None;
        }
        if let Some(index) = self.nodes.iter().position(|n| n.source == r) {
            return Some(index);
        }
        if self.nodes.len() >= MAX_NODES - 1 {
            return None;
        }

        let mask = self.width.mask();
        let (truth, children) = match *self.arena.op(r) {
            Op::Const(value) => ([value & mask; ASSIGNMENTS], 0),
            Op::Un(UnOp::Not, child) => {
                let index = self.visit(child)?;
                let node = &self.nodes[index];
                (node.truth.map(|value| !value & mask), node.reachable)
            }
            Op::Bin(op @ (BinOp::And | BinOp::Or | BinOp::Xor), a, b) => {
                let left = self.visit(a)?;
                let right = self.visit(b)?;
                let (left, right) = (&self.nodes[left], &self.nodes[right]);
                (
                    combine(op, &left.truth, &right.truth),
                    left.reachable | right.reachable,
                )
            }
            _ => {
                // This Ref, including the occurrence/generation in an Opaque,
                // Param or Load, is the identity. Never compare rendered tags.
                if self.atoms.len() == MAX_ATOMS {
                    return None;
                }
                let atom = self.atoms.len();
                self.atoms.push(r);
                (
                    std::array::from_fn(|assignment| {
                        if assignment & (1 << atom) == 0 {
                            0
                        } else {
                            mask
                        }
                    }),
                    0,
                )
            }
        };
        // Recursion may have consumed the remaining node budget.
        let index = self.nodes.len();
        if index >= MAX_NODES - 1 {
            return None;
        }
        self.nodes.push(ProvenNode {
            source: r,
            truth,
            reachable: children | (1 << index),
        });
        Some(index)
    }
}

fn combine(op: BinOp, left: &Truth, right: &Truth) -> Truth {
    std::array::from_fn(|i| match op {
        BinOp::And => left[i] & right[i],
        BinOp::Or => left[i] | right[i],
        BinOp::Xor => left[i] ^ right[i],
        _ => unreachable!("only bitwise operations enter the proof"),
    })
}

pub(super) fn simplify(arena: &mut Arena, op: BinOp, a: Ref, b: Ref, width: Width) -> Option<Ref> {
    if !matches!(op, BinOp::And | BinOp::Or | BinOp::Xor)
        || arena.width(a) != width
        || arena.width(b) != width
    {
        return None;
    }
    // Cheap gate before allocation; existing peepholes already handle a pair
    // of leaves. Simple arithmetic and the most common builders stop here.
    let nested = |r| {
        matches!(
            arena.op(r),
            Op::Bin(BinOp::And | BinOp::Or | BinOp::Xor, ..) | Op::Un(UnOp::Not, _)
        )
    };
    if !nested(a) && !nested(b) {
        return None;
    }
    let mut proof = Proof {
        arena,
        width,
        nodes: Vec::with_capacity(8),
        atoms: Vec::with_capacity(MAX_ATOMS),
        visits_left: MAX_VISITS,
    };
    let left = proof.visit(a)?;
    let right = proof.visit(b)?;
    let truth = combine(op, &proof.nodes[left].truth, &proof.nodes[right].truth);
    if truth.iter().all(|&value| value == truth[0]) {
        return Some(arena.constant(truth[0], width));
    }
    // Every collected source is a proper descendant of the prospective root.
    // Prefer the smallest opened DAG; Ref breaks ties deterministically.
    let best = proof
        .nodes
        .iter()
        .filter(|node| node.truth == truth)
        .min_by_key(|node| (node.reachable.count_ones(), node.source));
    let existing = best.map(|node| node.source);
    let old_local_cost = best.map_or(
        (proof.nodes[left].reachable | proof.nodes[right].reachable).count_ones() as usize + 1,
        |node| node.reachable.count_ones() as usize,
    );
    let Some(candidate) = canonical(&proof.atoms, &truth, width.mask()) else {
        return existing;
    };
    let roots = candidate.roots();
    if roots.len() + candidate.new_nodes() >= old_local_cost {
        return existing;
    }
    // An opaque atom can internally reference an opened Boolean node. Local
    // Boolean size alone therefore cannot establish a full reachable-DAG
    // decrease. Check full closures before creating any candidate nodes;
    // counting every proposed operation as new is a conservative upper bound.
    let old_full_cost = match existing {
        Some(r) => closure_len(arena, &[r]),
        None => closure_len(arena, &[a, b]).map(|count| count + 1),
    };
    let new_full_cost = closure_len(arena, &roots).map(|count| count + candidate.new_nodes());
    match (old_full_cost, new_full_cost) {
        (Some(old), Some(new)) if new < old => Some(candidate.build(arena, width)),
        _ => existing,
    }
}
