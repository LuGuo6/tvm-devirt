//! Unsigned index bounds implied by a path's Boolean predicates.
//! Every operation over-approximates possible indices. Unknown is the full
//! index domain; disjunction takes the union, never a convenient first arm.

use crate::ir::expr::{Arena, BinOp, Op, Ref};

fn union(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    }
}

fn intersect(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    Some(a?.min(b?))
}

/// Exact same-width affine identity: sign * base + offset, modulo the width.
/// This handles canonicalization of `(x + a) - b` to `x + (a-b)` without
/// confusing a range bound with a single value. Truncation is an opaque base.
fn affine(a: &Arena, r: Ref, width: crate::ir::expr::Width, depth: u32) -> (Ref, bool, u64) {
    if depth == 0 || a.width(r) != width { return (r, false, 0); }
    let mask = width.mask();
    match *a.op(r) {
        Op::Bin(BinOp::Add, x, y) => {
            let pair = a.as_const(y).map(|c| (x,c)).or_else(|| a.as_const(x).map(|c| (y,c)));
            if let Some((v,c)) = pair.filter(|(v,_)| a.width(*v) == width) {
                let (base, neg, offset) = affine(a,v,width,depth-1);
                return (base,neg,offset.wrapping_add(c)&mask);
            }
        }
        Op::Bin(BinOp::Sub, x, y) => {
            if let Some(c) = a.as_const(y).filter(|_| a.width(x) == width) {
                let (base,neg,offset) = affine(a,x,width,depth-1);
                return (base,neg,offset.wrapping_sub(c)&mask);
            }
            if let Some(c) = a.as_const(x).filter(|_| a.width(y) == width) {
                let (base,neg,offset) = affine(a,y,width,depth-1);
                return (base,!neg,c.wrapping_sub(offset)&mask);
            }
        }
        _ => {}
    }
    (r,false,0)
}

/// Bound on `idx` if `value == constant`. Only invert exact modular arithmetic
/// in idx's own width; projecting away high bits never binds those bits.
fn equal_bound(a: &Arena, value: Ref, constant: u64, idx: Ref, depth: u32) -> Option<u64> {
    let full = a.width(idx).mask();
    let mask = a.width(value).mask();
    if constant & !mask != 0 { return None; }
    if value == idx { return Some(constant); }
    if depth == 0 { return Some(full); }
    if let Some(c) = a.as_const(value) {
        return (c == constant).then_some(full);
    }
    if a.width(value) == a.width(idx) {
        let (vb,vneg,voff) = affine(a,value,a.width(idx),depth);
        let (ib,ineg,ioff) = affine(a,idx,a.width(idx),depth);
        if vb == ib {
            let base = if vneg { voff.wrapping_sub(constant) }
                else { constant.wrapping_sub(voff) };
            return Some((if ineg { ioff.wrapping_sub(base) }
                else { base.wrapping_add(ioff) }) & full);
        }
    }
    match *a.op(value) {
        Op::Bin(BinOp::Add, x, y) if a.width(value) == a.width(idx) => {
            if let Some(c) = a.as_const(y) {
                equal_bound(a, x, constant.wrapping_sub(c) & mask, idx, depth - 1)
            } else if let Some(c) = a.as_const(x) {
                equal_bound(a, y, constant.wrapping_sub(c) & mask, idx, depth - 1)
            } else { Some(full) }
        }
        Op::Bin(BinOp::Sub, x, y) if a.width(value) == a.width(idx) => {
            if let Some(c) = a.as_const(y) {
                equal_bound(a, x, constant.wrapping_add(c) & mask, idx, depth - 1)
            } else if let Some(c) = a.as_const(x) {
                equal_bound(a, y, c.wrapping_sub(constant) & mask, idx, depth - 1)
            } else { Some(full) }
        }
        Op::Zext(x) => equal_bound(a, x, constant, idx, depth - 1),
        Op::Sext(x) => {
            let bits = a.width(x).bits();
            let narrow = constant & a.width(x).mask();
            let extended = ((narrow << (64-bits)) as i64 >> (64-bits)) as u64 & mask;
            if extended != constant { None }
            else { equal_bound(a, x, narrow, idx, depth - 1) }
        }
        _ => Some(full),
    }
}

/// The greatest possible idx, or None for an impossible condition. This proves
/// only a bound; concrete enumeration still checks each target independently.
pub(super) fn condition_bound(a: &Arena, value: Ref, nonzero: bool, idx: Ref) -> Option<u64> {
    fn go(a: &Arena, value: Ref, nonzero: bool, idx: Ref, depth: u32) -> Option<u64> {
        let full = a.width(idx).mask();
        if let Some(c) = a.as_const(value) { return ((c != 0) == nonzero).then_some(full); }
        if value == idx { return Some(if nonzero { full } else { 0 }); }
        if depth == 0 { return Some(full); }
        let recur = |v, nz| go(a, v, nz, idx, depth - 1);
        match *a.op(value) {
            Op::Zext(x) | Op::Sext(x) => recur(x, nonzero),
            // A nonzero projection proves the original was nonzero. The reverse
            // inference is invalid when discarded high bits might be set.
            Op::Trunc(x) if nonzero => recur(x, true),
            Op::Bin(BinOp::Shl | BinOp::Shr | BinOp::Sar, x, _) if nonzero => recur(x, true),
            Op::Bin(BinOp::Or, x, y) => {
                if nonzero { union(recur(x, true), recur(y, true)) }
                else { intersect(recur(x, false), recur(y, false)) }
            }
            Op::Bin(BinOp::And, x, y) if nonzero => intersect(recur(x, true), recur(y, true)),
            Op::Bin(BinOp::Eq, x, y) => {
                if let Some(c) = a.as_const(y) {
                    if nonzero { equal_bound(a, x, c, idx, depth - 1) }
                    else if c == 0 { recur(x, true) }
                    else { Some(full) }
                } else if let Some(c) = a.as_const(x) {
                    if nonzero { equal_bound(a, y, c, idx, depth - 1) }
                    else if c == 0 { recur(y, true) }
                    else { Some(full) }
                } else { Some(full) }
            }
            Op::Bin(BinOp::Ult, x, y) => {
                if x == idx && nonzero {
                    a.as_const(y).map_or(Some(full), |c| c.checked_sub(1))
                } else if y == idx && !nonzero {
                    Some(a.as_const(x).unwrap_or(full))
                } else { Some(full) }
            }
            _ if !nonzero => equal_bound(a, value, 0, idx, depth - 1),
            _ => Some(full),
        }
    }
    go(a, value, nonzero, idx, 32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::expr::{Reg, Width};

    fn eval(a: &Arena, r: Ref, x: u64, y: u64) -> u64 {
        let w = a.width(r);
        let v = match *a.op(r) {
            Op::Const(c) => c,
            Op::InitReg(Reg::Rdx) => x,
            Op::InitReg(Reg::Rcx) => y,
            Op::Bin(op, l, r) => crate::ir::expr::fold_bin(
                op, eval(a, l, x, y), eval(a, r, x, y), a.width(l).max(a.width(r))
            ).expect("fixture operation"),
            Op::Trunc(v) | Op::Zext(v) => eval(a, v, x, y),
            Op::Sext(v) => {
                let shift = 64 - a.width(v).bits();
                ((eval(a, v, x, y) << shift) as i64 >> shift) as u64
            }
            ref other => panic!("unhandled fixture {other:?}"),
        };
        v & w.mask()
    }

    #[test]
    fn bounds_overapproximate_exhaustive_byte_predicates() {
        let mut a = Arena::new();
        let x = a.init_reg(Reg::Rdx);
        let y = a.init_reg(Reg::Rcx);
        let idx = a.trunc(x, Width::W8);
        let other = a.trunc(y, Width::W8);
        let mut checks = vec![];
        for c in [0, 1, 3, 7, 63, 64, 127, 128, 254, 255] {
            let k = a.constant(c, Width::W8);
            let lt = a.bin(BinOp::Ult, idx, k);
            let gt = a.bin(BinOp::Ult, k, idx);
            let eq = a.bin(BinOp::Eq, idx, k);
            let sum = a.bin(BinOp::Add, idx, k);
            let difference = a.bin(BinOp::Sub, k, idx);
            let z = a.constant(0, Width::W8);
            let sum_zero = a.bin(BinOp::Eq, sum, z);
            let diff_zero = a.bin(BinOp::Eq, difference, z);
            let le = a.bin(BinOp::Or, lt, eq);
            let either = a.bin(BinOp::Or, le, other);
            let both = a.bin(BinOp::And, le, other);
            for predicate in [lt, gt, eq, sum_zero, diff_zero, le, either, both] {
                for nonzero in [false, true] {
                    checks.push((predicate, nonzero, condition_bound(&a, predicate, nonzero, idx)));
                }
            }
        }
        for x in 0..=255 {
            for y in 0..=255 {
                for &(p, nz, bound) in &checks {
                    if (eval(&a, p, x, y) != 0) == nz {
                        assert!(bound.is_some_and(|b| x <= b), "x={x} y={y} p={p:?} nz={nz} bound={bound:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn wide_projection_and_sign_extension_keep_reachable_high_values() {
        let mut a = Arena::new();
        let x = a.init_reg(Reg::Rdx);
        for width in [Width::W16, Width::W32, Width::W64] {
            let idx = a.trunc(x, width);
            let low = a.trunc(idx, Width::W8);
            let zero = a.constant(0, Width::W8);
            let low_zero = a.bin(BinOp::Eq, low, zero);
            assert_eq!(condition_bound(&a, low_zero, true, idx), Some(width.mask()));
            let sx = a.sext(low, width);
            let k = a.constant(0xff, width);
            let impossible = a.bin(BinOp::Eq, sx, k);
            assert_eq!(condition_bound(&a, impossible, true, idx), None);
            let minus1 = a.constant(width.mask(), width);
            let valid = a.bin(BinOp::Eq, sx, minus1);
            assert_eq!(condition_bound(&a, valid, true, idx), Some(width.mask()));
        }
    }

    #[test]
    fn unsigned_bounds_do_not_treat_signed_upper_range_as_small() {
        let mut a = Arena::new();
        let r = a.init_reg(Reg::Rdx);
        let idx = a.trunc(r, Width::W32);
        let n = a.constant(7, Width::W32);
        let lt = a.bin(BinOp::Ult, idx, n);
        assert_eq!(condition_bound(&a, lt, true, idx), Some(6));
        assert_eq!(condition_bound(&a, lt, false, idx), Some(u32::MAX as u64));
        let slt = a.bin(BinOp::Slt, idx, n);
        assert_eq!(condition_bound(&a, slt, true, idx), Some(u32::MAX as u64));
    }

    #[test]
    fn equality_inverts_wrapping_offset_without_losing_top_bits() {
        let mut a = Arena::new();
        let r = a.init_reg(Reg::Rdx);
        let idx = a.trunc(r, Width::W32);
        let minus4 = a.constant(0xffff_fffc, Width::W32);
        let difference = a.bin(BinOp::Add, idx, minus4);
        let zero = a.constant(0, Width::W32);
        let eq = a.bin(BinOp::Eq, difference, zero);
        assert_eq!(condition_bound(&a, eq, true, idx), Some(4));
        assert_eq!(condition_bound(&a, eq, false, idx), Some(u32::MAX as u64));
        let low = a.trunc(idx, Width::W8);
        let z8 = a.constant(0, Width::W8);
        let eq8 = a.bin(BinOp::Eq, low, z8);
        assert_eq!(condition_bound(&a, eq8, true, idx), Some(u32::MAX as u64));
    }

    #[test]
    fn canonicalized_affine_equality_retains_exact_offset_relation() {
        let mut a = Arena::new();
        let r = a.init_reg(Reg::Rdx);
        let narrow = a.trunc(r, Width::W32);
        let bias = a.constant(0xffedffff, Width::W32);
        let idx = a.bin(BinOp::Add, narrow, bias);
        let zf_bias = a.constant(0xffedffde, Width::W32);
        let shifted = a.bin(BinOp::Add, narrow, zf_bias);
        let zero = a.constant(0, Width::W32);
        let zf = a.bin(BinOp::Eq, shifted, zero);
        assert_eq!(condition_bound(&a,zf,true,idx),Some(0x21));
        let other = a.init_reg(Reg::Rcx);
        let other32 = a.trunc(other, Width::W32);
        let unknown = a.bin(BinOp::Eq, other32, zero);
        assert_eq!(condition_bound(&a,unknown,true,idx),Some(u32::MAX as u64));
        let low = a.trunc(narrow,Width::W8);
        let projected = a.zext(low,Width::W32);
        let projected_zero = a.bin(BinOp::Eq,projected,zero);
        assert_eq!(condition_bound(&a,projected_zero,true,idx),Some(u32::MAX as u64));
    }

    #[test]
    fn unknown_disjunction_and_shift_loss_keep_full_domain() {
        let mut a = Arena::new();
        let idx = a.init_reg(Reg::Rdx);
        let other = a.init_reg(Reg::Rcx);
        let n = a.constant(5, Width::W64);
        let p = a.bin(BinOp::Ult, idx, n);
        let p64 = a.zext(p, Width::W64);
        let both = a.bin(BinOp::Or, p64, other);
        assert_eq!(condition_bound(&a, both, true, idx), Some(u64::MAX));
        let shift = a.constant(63, Width::W64);
        let lossy = a.bin(BinOp::Shl, idx, shift);
        assert_eq!(condition_bound(&a, lossy, false, idx), Some(u64::MAX));
    }
}
