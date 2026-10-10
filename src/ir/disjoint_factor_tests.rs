use super::*;

#[test]
fn single_leaf_and_complementary_mask_have_no_carry() {
    for width in [Width::W8, Width::W16, Width::W32, Width::W64] {
        let mut arena = Arena::new();
        let input = arena.init_reg(Reg::Rax);
        let x = arena.trunc(input, width);
        let four = arena.constant(4, width);
        let q4 = arena.bin(BinOp::Mul, x, four);
        let complement = arena.un(UnOp::Not, q4);
        let masked = arena.bin(BinOp::And, x, complement);
        assert!(arena.disjoint_by_factors(q4, masked));
        assert!(arena.disjoint_by_factors(masked, q4));
        let sum = arena.bin(BinOp::Add, q4, masked);
        let selected = arena.bin(BinOp::And, q4, sum);
        assert_eq!(selected, q4, "{}", render(&arena, selected, 12));
        for value in (0..=255u64).chain([0x123456789abcdef0, u64::MAX, 1 << 63]) {
            let env = HashMap::from([(Reg::Rax, value)]);
            let x = value & width.mask();
            let q4 = x.wrapping_mul(4) & width.mask();
            assert_eq!(tests::eval(&arena, sum, &env), q4.wrapping_add(x & !q4) & width.mask());
            assert_eq!(tests::eval(&arena, selected, &env), q4);
        }
    }
}

#[test]
fn complementary_factor_proof_requires_same_width_and_identity() {
    let mut arena = Arena::new();
    let wide = arena.init_reg(Reg::Rax);
    let other = arena.init_reg(Reg::Rcx);
    let narrow = arena.trunc(wide, Width::W8);
    let complement = arena.un(UnOp::Not, wide);
    let masked = arena.bin(BinOp::And, other, complement);
    assert!(!arena.disjoint_by_factors(narrow, masked));
    let complement = arena.un(UnOp::Not, other);
    let masked = arena.bin(BinOp::And, wide, complement);
    assert!(!arena.disjoint_by_factors(wide, masked));
}
