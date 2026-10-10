//! Independent fixed-width proof and builder-order review for union subtraction.
use super::*;

// Opaque occurrences are separately assignable. The existing sampling evaluator
// gives every opaque zero, which would conceal the identity-negative cases here.
fn evaluate(a: &Arena, r: Ref, regs: &HashMap<Reg, u64>, opaque: &HashMap<Ref, u64>) -> u64 {
    let value = match *a.op(r) {
        Op::Const(v) => v,
        Op::InitReg(reg) => regs[&reg],
        Op::Opaque(..) => opaque[&r],
        Op::Trunc(p) | Op::Zext(p) => evaluate(a, p, regs, opaque),
        Op::Un(UnOp::Not, p) => !evaluate(a, p, regs, opaque),
        Op::Un(UnOp::Neg, p) => evaluate(a, p, regs, opaque).wrapping_neg(),
        Op::Bin(op, p, q) => {
            let x = evaluate(a, p, regs, opaque);
            let y = evaluate(a, q, regs, opaque);
            match op {
                BinOp::Add => x.wrapping_add(y),
                BinOp::Sub => x.wrapping_sub(y),
                BinOp::Mul => x.wrapping_mul(y),
                BinOp::And => x & y,
                BinOp::Or => x | y,
                BinOp::Xor => x ^ y,
                BinOp::UDiv => x / y,
                BinOp::URem => x % y,
                _ => panic!("Unexpected operation in union-sub review: {op:?}"),
            }
        }
        ref op => panic!("Unexpected node in union-sub review: {op:?}"),
    };
    value & a.width(r).mask()
}

#[test]
fn union_sub_scalar_exhausts_all_byte_pairs_without_borrow() {
    for x in 0..=255u64 {
        for y in 0..=255u64 {
            let union = x | y;
            let removed = union.wrapping_sub(y) & 255;
            assert_eq!(removed, x & !y);
            assert_eq!(removed & y, 0);
            assert_eq!(removed + y, union);
            assert_eq!(union ^ removed, y);
        }
    }
}

#[test]
fn union_sub_symbolic_same_width_builds_the_exact_complement() {
    for width in [Width::W8, Width::W16, Width::W32, Width::W64] {
        let mut a = Arena::new();
        let rx = a.init_reg(Reg::Rax);
        let ry = a.init_reg(Reg::Rcx);
        let x = a.trunc(rx, width);
        let y = a.trunc(ry, width);
        for (left, right) in [(x, y), (y, x)] {
            let union = a.bin(BinOp::Or, left, right);
            let got = a.bin(BinOp::Sub, union, right);
            let not_right = a.not(right);
            let expected = a.bin(BinOp::And, left, not_right);
            assert_eq!(got, expected, "{}", render(&a, got, 12));
            assert_eq!(a.width(got), width);
        }
    }
}

#[test]
fn union_xor_sub_cancels_both_raw_and_eagerly_simplified_children() {
    for width in [Width::W8, Width::W16, Width::W32, Width::W64] {
        let mut a = Arena::new();
        let rx = a.init_reg(Reg::Rax);
        let ry = a.init_reg(Reg::Rcx);
        let x = a.trunc(rx, width);
        let y = a.trunc(ry, width);
        for (left, right) in [(x, y), (y, x)] {
            let union = a.bin(BinOp::Or, left, right);
            let raw_sub = a.intern(Node { op: Op::Bin(BinOp::Sub, union, right), width });
            let eager_sub = a.bin(BinOp::Sub, union, right);
            for subtraction in [raw_sub, eager_sub] {
                for (p, q) in [(union, subtraction), (subtraction, union)] {
                    let got = a.bin(BinOp::Xor, p, q);
                    assert_eq!(got, right, "{}", render(&a, got, 12));
                    assert_eq!(a.width(got), width);
                }
            }
        }
    }
}

#[test]
fn decimal_union_sub_dag_is_urem_and_exhausts_u16_inputs() {
    for width in [Width::W8, Width::W16, Width::W32, Width::W64] {
        let mut a = Arena::new();
        let input = a.init_reg(Reg::Rdx);
        let n = a.trunc(input, width);
        let ten = a.constant(10, width);
        let q = a.bin(BinOp::UDiv, n, ten);
        let four = a.constant(4, width);
        let q4 = a.bin(BinOp::Mul, q, four);
        let union = a.bin(BinOp::Or, q, q4);
        let removed = a.bin(BinOp::Sub, union, q4);
        let recovered_q4 = a.bin(BinOp::Xor, union, removed);
        let q5 = a.bin(BinOp::Add, q, recovered_q4);
        let two = a.constant(2, width);
        let q10 = a.bin(BinOp::Mul, q5, two);
        let got = a.bin(BinOp::Sub, n, q10);
        let expected = a.bin(BinOp::URem, n, ten);
        assert_eq!(got, expected, "{}", render(&a, got, 20));
        assert_eq!(a.width(got), width);
        assert_eq!(a.known_zero(got) & width.mask() & !15, width.mask() & !15);
        let count = if width == Width::W8 { 255 } else { 65535 };
        for value in (0..=count).chain([
            width.sign_bit(), width.mask() / 2, width.mask() - 10,
            width.mask() - 1, width.mask(),
        ]) {
            let env = HashMap::from([(Reg::Rdx, value)]);
            let n = value & width.mask();
            let quotient = n / 10;
            assert_eq!(evaluate(&a, got, &env, &HashMap::new()), n % 10);
            // These concrete bounds justify the unsigned mathematical step;
            // no division/MBA rule is inferred from target addresses or values.
            assert!(quotient <= width.mask() / 10);
            assert_eq!(n.wrapping_sub((quotient + quotient * 4) * 2), n % 10);
        }
    }
}

#[test]
fn mixed_width_complement_must_not_clear_the_wide_high_bits() {
    let mut a = Arena::new();
    let rx = a.init_reg(Reg::Rax);
    let ry = a.init_reg(Reg::Rcx);
    let x = a.trunc(rx, Width::W16);
    let y = a.trunc(ry, Width::W8);
    let union = a.bin(BinOp::Or, x, y);
    let got = a.bin(BinOp::Sub, union, y);
    let not_y8 = a.not(y);
    let unsound_narrow_complement = a.bin(BinOp::And, x, not_y8);
    let env = HashMap::from([(Reg::Rax, 0x100), (Reg::Rcx, 0)]);
    assert_eq!(evaluate(&a, got, &env, &HashMap::new()), 0x100);
    assert_eq!(evaluate(&a, unsound_narrow_complement, &env, &HashMap::new()), 0);
    assert_ne!(got, unsound_narrow_complement);
    let outer = a.bin(BinOp::Xor, union, got);
    assert_eq!(a.width(outer), Width::W16);
    assert_ne!(outer, y, "Returning W8 Y would silently narrow a W16 result");
}

#[test]
fn truncated_union_loses_the_subset_proof_for_wide_subtraction() {
    let mut a = Arena::new();
    let rx = a.init_reg(Reg::Rax);
    let ry = a.init_reg(Reg::Rcx);
    let x = a.trunc(rx, Width::W16);
    let y = a.trunc(ry, Width::W16);
    let union16 = a.bin(BinOp::Or, x, y);
    let union8 = a.trunc(union16, Width::W8);
    let subtraction = a.bin(BinOp::Sub, union8, y);
    let outer = a.bin(BinOp::Xor, union8, subtraction);
    let env = HashMap::from([(Reg::Rax, 0), (Reg::Rcx, 0x100)]);
    assert_eq!(evaluate(&a, outer, &env, &HashMap::new()), 0xff00);
    assert_ne!(outer, y);
    assert_eq!(a.width(outer), Width::W16);
}

#[test]
fn non_contained_subtrahend_keeps_the_borrow_dependency() {
    let mut a = Arena::new();
    let rx = a.init_reg(Reg::Rax);
    let ry = a.init_reg(Reg::Rcx);
    let rz = a.init_reg(Reg::Rdx);
    let x = a.trunc(rx, Width::W8);
    let y = a.trunc(ry, Width::W8);
    let z = a.trunc(rz, Width::W8);
    let union = a.bin(BinOp::Or, x, y);
    let subtraction = a.bin(BinOp::Sub, union, z);
    let outer = a.bin(BinOp::Xor, union, subtraction);
    let env = HashMap::from([(Reg::Rax, 1), (Reg::Rcx, 2), (Reg::Rdx, 4)]);
    assert_eq!(evaluate(&a, outer, &env, &HashMap::new()), 0xfc);
    assert_ne!(outer, y);
    assert_ne!(outer, z);
}

#[test]
fn opaque_same_tag_distinct_identity_must_not_cancel() {
    for width in [Width::W8, Width::W16, Width::W32, Width::W64] {
        let mut a = Arena::new();
        let x = a.opaque("union-input", width);
        let first = a.opaque("same-tag", width);
        let second = a.opaque("same-tag", width);
        assert_ne!(first, second);
        let union = a.bin(BinOp::Or, x, first);
        let subtraction = a.bin(BinOp::Sub, union, second);
        let got = a.bin(BinOp::Xor, union, subtraction);
        let values = HashMap::from([(x, 0), (first, 1), (second, 2)]);
        assert_eq!(evaluate(&a, got, &HashMap::new(), &values), width.mask() - 1);
        assert_ne!(got, first);
        assert_ne!(got, second);
        let dependencies = leaves(&a, got);
        assert!(dependencies.contains(&first) && dependencies.contains(&second));
        let exact_subtraction = a.bin(BinOp::Sub, union, first);
        let exact = a.bin(BinOp::Xor, union, exact_subtraction);
        assert_eq!(exact, first);
    }
}
