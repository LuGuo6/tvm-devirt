//! Independent boundary and byte-domain checks for common-OR subtraction.
use super::*;

#[test]
fn all_byte_inputs_and_masks_agree_with_minimal_known_bit_proof() {
    // Exhaust all 16,777,216 triples. Only differing mask bits need to be
    // known; the remaining input bits are deliberately absent from the proof.
    for x in 0..=255u64 {
        for c1 in 0..=255u64 {
            for c2 in 0..=255u64 {
                let required = c1 ^ c2;
                let known_ones = x & required;
                let proof_value = (c1 | known_ones).wrapping_sub(c2 | known_ones) & 0xff;
                let actual = (x | c1).wrapping_sub(x | c2) & 0xff;
                assert_eq!(proof_value, actual);
            }
        }
    }
}

#[test]
fn every_byte_mask_pair_matches_actual_expression_with_partial_known_bits() {
    for (unknown, forced) in [(0x55u64, 0x80u64), (0xaau64, 0x01u64)] {
        let values: Vec<_> = (0..=255u64).filter(|v| v & !unknown == 0).collect();
        for c1 in 0..=255u64 {
            for c2 in 0..=255u64 {
                let mut arena = Arena::new();
                let input = arena.init_reg(Reg::Rax);
                let input = arena.trunc(input, Width::W8);
                let allowed = arena.constant(unknown, Width::W8);
                let x = arena.bin(BinOp::And, input, allowed);
                let fixed = arena.constant(forced, Width::W8);
                let x = arena.bin(BinOp::Or, x, fixed);
                let m1 = arena.constant(c1, Width::W8);
                let m2 = arena.constant(c2, Width::W8);
                let a = arena.bin(BinOp::Or, x, m1);
                let b = arena.bin(BinOp::Or, x, m2);
                let got = arena.bin(BinOp::Sub, a, b);
                for &value in &values {
                    let env = HashMap::from([(Reg::Rax, value)]);
                    let x = value | forced;
                    assert_eq!(tests::eval(&arena, got, &env), (x | c1).wrapping_sub(x | c2) & 0xff,
                        "unknown={unknown:x}, forced={forced:x}, masks={c1:x}/{c2:x}, input={value:x}");
                }
            }
        }
    }
}

#[test]
fn differing_unknown_bit_retains_two_observable_results() {
    let mut arena = Arena::new();
    let input = arena.init_reg(Reg::Rax);
    let input = arena.trunc(input, Width::W8);
    let one = arena.constant(1, Width::W8);
    let x = arena.bin(BinOp::And, input, one);
    let c1 = arena.constant(0x20, Width::W8);
    let c2 = arena.constant(0x21, Width::W8);
    let a = arena.bin(BinOp::Or, x, c1);
    let b = arena.bin(BinOp::Or, x, c2);
    let difference = arena.bin(BinOp::Sub, a, b);
    assert!(arena.as_const(difference).is_none());
    assert_eq!(tests::eval(&arena, difference, &HashMap::from([(Reg::Rax, 0)])), 0xff);
    assert_eq!(tests::eval(&arena, difference, &HashMap::from([(Reg::Rax, 1)])), 0);
}

#[test]
fn narrowing_only_one_operand_keeps_upper_input_dependency() {
    let mut arena = Arena::new();
    let input = arena.init_reg(Reg::Rax);
    let input = arena.trunc(input, Width::W16);
    let mask = arena.constant(0x110, Width::W16);
    let x = arena.bin(BinOp::And, input, mask);
    let c1 = arena.constant(2, Width::W16);
    let c2 = arena.constant(0xc2, Width::W16);
    let a = arena.bin(BinOp::Or, x, c1);
    let a = arena.trunc(a, Width::W8);
    let b = arena.bin(BinOp::Or, x, c2);
    let difference = arena.bin(BinOp::Sub, a, b);
    assert!(arena.as_const(difference).is_none());
    for value in [0, 0x10, 0x100, 0x110] {
        let env = HashMap::from([(Reg::Rax, value)]);
        let expected = ((value | 2u64) & 0xff).wrapping_sub(value | 0xc2u64) & 0xffff;
        assert_eq!(tests::eval(&arena, difference, &env), expected);
    }
}
