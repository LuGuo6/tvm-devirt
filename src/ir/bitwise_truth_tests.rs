//! Regression fixtures for bounded, same-width Boolean absorption.
use super::*;

fn raw_bin(a: &mut Arena, op: BinOp, x: Ref, y: Ref, width: Width) -> Ref {
    a.intern(Node {
        op: Op::Bin(op, x, y),
        width,
    })
}

fn raw_not(a: &mut Arena, x: Ref, width: Width) -> Ref {
    a.intern(Node {
        op: Op::Un(UnOp::Not, x),
        width,
    })
}

#[test]
fn bitwise_truth_observed_decimal_union_absorbs_xor() {
    // 0x1812349a0, decimal-union-diagnostic/summary.json: the divisor
    // multiplier includes (Q & 4Q) + ((Q ^ 4Q) | (Q | 4Q)).
    let mut a = Arena::new();
    let x = a.init_reg(Reg::Rdx);
    let ten = a.constant(10, Width::W64);
    let q = a.bin(BinOp::UDiv, x, ten);
    let four = a.constant(4, Width::W64);
    let q4 = a.bin(BinOp::Mul, q, four);
    let intersection = a.bin(BinOp::And, q, q4);
    let xor = a.bin(BinOp::Xor, q, q4);
    let union = a.bin(BinOp::Or, q, q4);
    let redundant = a.bin(BinOp::Or, xor, union);
    assert_eq!(redundant, union, "{}", render(&a, redundant, 20));
    let five_q = a.bin(BinOp::Add, intersection, redundant);
    let two = a.constant(2, Width::W64);
    let ten_q = a.bin(BinOp::Mul, five_q, two);
    let remainder = a.bin(BinOp::Sub, x, ten_q);
    let expected = a.bin(BinOp::URem, x, ten);
    assert_eq!(remainder, expected, "{}", render(&a, remainder, 20));
}

#[test]
fn bitwise_truth_observed_pointer_partition_cancels() {
    // 0x1811c7ff0: ~( (~rdx & 0x10) ^ (~rdx & ~0x10) ).
    for width in [Width::W8, Width::W16, Width::W32, Width::W64] {
        let mut a = Arena::new();
        let reg = a.init_reg(Reg::Rdx);
        let x = a.trunc(reg, width);
        let nx = a.not(x);
        let low = a.constant(0x10, width);
        let high = a.constant(!0x10, width);
        let left = a.bin(BinOp::And, nx, low);
        let right = a.bin(BinOp::And, nx, high);
        let partition = a.bin(BinOp::Xor, left, right);
        let restored = a.not(partition);
        assert_eq!(restored, x, "{}", render(&a, restored, 20));
    }
}

#[test]
fn bitwise_truth_nonuniform_constant_is_proved_per_bit() {
    for width in [Width::W8, Width::W16, Width::W32, Width::W64] {
        let mut a = Arena::new();
        let rx = a.init_reg(Reg::Rax);
        let ry = a.init_reg(Reg::Rcx);
        let x = a.trunc(rx, width);
        let y = a.trunc(ry, width);
        let c = a.constant(0x8142_2418_a55a_c33c, width);
        let nc = a.constant(!0x8142_2418_a55a_c33c, width);
        let xc = raw_bin(&mut a, BinOp::Or, x, c, width);
        let yc = raw_bin(&mut a, BinOp::Or, y, c, width);
        let left = raw_bin(&mut a, BinOp::And, xc, yc, width);
        let xy = raw_bin(&mut a, BinOp::And, x, y, width);
        let right = raw_bin(&mut a, BinOp::And, xy, nc, width);
        let result = a.bin(BinOp::Xor, left, right);
        assert_eq!(result, c, "{}", render(&a, result, 20));
    }
}

#[test]
fn bitwise_truth_four_atom_shared_dag_cancels() {
    let mut a = Arena::new();
    let [x, y, z, w] = [Reg::Rax, Reg::Rcx, Reg::Rdx, Reg::Rbx].map(|r| a.init_reg(r));
    let xy = raw_bin(&mut a, BinOp::Xor, x, y, Width::W64);
    let zw = raw_bin(&mut a, BinOp::Xor, z, w, Width::W64);
    let left = raw_bin(&mut a, BinOp::Xor, xy, zw, Width::W64);
    let xyz = raw_bin(&mut a, BinOp::Xor, xy, z, Width::W64);
    let right = raw_bin(&mut a, BinOp::Xor, xyz, w, Width::W64);
    let result = a.bin(BinOp::Xor, left, right);
    assert_eq!(a.as_const(result), Some(0), "{}", render(&a, result, 20));
}

// Independent evaluator: do not call fold_bin or the proof's truth evaluator.
// Opaque occurrences remain separately assignable, including identical tags.
fn evaluate(a: &Arena, r: Ref, inputs: &HashMap<Ref, u64>) -> u64 {
    let value = if let Some(&value) = inputs.get(&r) {
        value
    } else {
        match *a.op(r) {
            Op::Const(value) => value,
            Op::Un(UnOp::Not, x) => !evaluate(a, x, inputs),
            Op::Bin(op, x, y) => {
                let (x, y) = (evaluate(a, x, inputs), evaluate(a, y, inputs));
                match op {
                    BinOp::And => x & y,
                    BinOp::Or => x | y,
                    BinOp::Xor => x ^ y,
                    BinOp::Add => x.wrapping_add(y),
                    BinOp::Sub => x.wrapping_sub(y),
                    BinOp::Mul => x.wrapping_mul(y),
                    _ => panic!("unexpected test operation {op:?}"),
                }
            }
            ref other => panic!("unassigned test atom {r:?}: {other:?}"),
        }
    };
    value & a.width(r).mask()
}

fn absorption_fixtures(a: &mut Arena, x: Ref, y: Ref, width: Width) -> Vec<(Ref, Ref)> {
    let xor = raw_bin(a, BinOp::Xor, x, y, width);
    let union = raw_bin(a, BinOp::Or, x, y, width);
    let common = raw_bin(a, BinOp::And, x, y, width);
    let nx = raw_not(a, x, width);
    let ny = raw_not(a, y, width);
    let not_common = raw_not(a, common, width);
    let not_union = raw_not(a, union, width);
    let complement_xor = raw_bin(a, BinOp::Xor, nx, ny, width);
    let k = a.constant(0x8142_2418_a55a_c33c, width);
    let nk = a.constant(!0x8142_2418_a55a_c33c, width);
    let xk = raw_bin(a, BinOp::Or, x, k, width);
    let yk = raw_bin(a, BinOp::Or, y, k, width);
    let forced = raw_bin(a, BinOp::And, xk, yk, width);
    let residual = raw_bin(a, BinOp::And, common, nk, width);
    let low = raw_bin(a, BinOp::And, nx, k, width);
    let high = raw_bin(a, BinOp::And, nx, nk, width);
    let recipes = [
        (BinOp::Or, xor, union),
        (BinOp::And, xor, not_common),
        (BinOp::And, xor, not_union),
        (BinOp::Xor, xor, complement_xor),
        (BinOp::Xor, forced, residual),
        (BinOp::Xor, low, high),
    ];
    recipes
        .into_iter()
        .map(|(op, left, right)| {
            let original = raw_bin(a, op, left, right, width);
            let simplified = bitwise_truth::simplify(a, op, left, right, width)
                .expect("fixture must exercise the bounded proof");
            assert_ne!(original, simplified);
            (original, simplified)
        })
        .collect()
}

#[test]
fn bitwise_truth_replacements_match_every_byte_pair() {
    let mut a = Arena::new();
    let x = a.opaque("x", Width::W8);
    let y = a.opaque("y", Width::W8);
    let fixtures = absorption_fixtures(&mut a, x, y, Width::W8);
    for vx in 0..=255 {
        for vy in 0..=255 {
            let inputs = HashMap::from([(x, vx), (y, vy)]);
            for &(original, simplified) in &fixtures {
                assert_eq!(
                    evaluate(&a, original, &inputs),
                    evaluate(&a, simplified, &inputs)
                );
            }
        }
    }
}

#[test]
fn bitwise_truth_replacements_match_wide_random_and_boundary_values() {
    let mut seed = 0x8fed_a03a_1724_731bu64;
    for width in [Width::W16, Width::W32, Width::W64] {
        let mut a = Arena::new();
        let x = a.opaque("x", width);
        let y = a.opaque("y", width);
        let fixtures = absorption_fixtures(&mut a, x, y, width);
        let mut values = vec![0, 1, 2, width.mask(), width.mask() >> 1, width.sign_bit()];
        values.extend([
            0xaaaa_aaaa_aaaa_aaaa,
            0x5555_5555_5555_5555,
            0x8000_0000_0000_0001,
        ]);
        for _ in 0..4096 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            values.push(seed);
        }
        for (i, &vx) in values.iter().enumerate() {
            // Different and repeated lanes/values, not just all-zero/all-one atoms.
            let vy = values[(i * 17 + 7) % values.len()];
            let inputs = HashMap::from([(x, vx), (y, vy)]);
            for &(original, simplified) in &fixtures {
                assert_eq!(
                    evaluate(&a, original, &inputs),
                    evaluate(&a, simplified, &inputs)
                );
            }
        }
    }
}

#[test]
fn bitwise_truth_rejects_mixed_width_networks_and_non_bitwise_roots() {
    let mut a = Arena::new();
    let x = a.opaque("x", Width::W64);
    let y = a.opaque("y", Width::W8);
    let mixed = raw_bin(&mut a, BinOp::Or, x, y, Width::W64);
    assert_eq!(
        bitwise_truth::simplify(&mut a, BinOp::And, mixed, x, Width::W64),
        None
    );
    assert_eq!(
        bitwise_truth::simplify(&mut a, BinOp::And, mixed, y, Width::W64),
        None
    );
    let z = a.opaque("z", Width::W64);
    let nested = raw_bin(&mut a, BinOp::Or, x, z, Width::W64);
    assert_eq!(
        bitwise_truth::simplify(&mut a, BinOp::Add, nested, x, Width::W64),
        None
    );
    let sum = raw_bin(&mut a, BinOp::Add, x, z, Width::W64);
    assert_eq!(
        bitwise_truth::simplify(&mut a, BinOp::Xor, sum, nested, Width::W64),
        None
    );
}

#[test]
fn bitwise_truth_preserves_opaque_load_and_param_occurrence_identity() {
    let mut a = Arena::new();
    let x = a.opaque("x", Width::W64);
    let oa = a.opaque("same tag", Width::W64);
    let ob = a.opaque("same tag", Width::W64);
    let address = a.constant(0x180010000, Width::W64);
    let la = a.load(address, Width::W64);
    a.bump_mem_gen();
    a.note_symbolic_store();
    let lb = a.load(address, Width::W64);
    let pa = a.param(BlockRef(1), Reg::Rax);
    let pb = a.param(BlockRef(2), Reg::Rax);
    let nx = raw_not(&mut a, x, Width::W64);
    for (left, right) in [(oa, ob), (la, lb), (pa, pb)] {
        assert_ne!(left, right);
        let difference = raw_bin(&mut a, BinOp::Xor, left, right, Width::W64);
        let residue = raw_bin(&mut a, BinOp::And, difference, nx, Width::W64);
        assert_eq!(
            bitwise_truth::simplify(&mut a, BinOp::Or, x, residue, Width::W64),
            None
        );
        let original = raw_bin(&mut a, BinOp::Or, x, residue, Width::W64);
        let inputs = HashMap::from([(x, 0), (left, 1), (right, 0)]);
        assert_eq!(evaluate(&a, original, &inputs), 1);
    }
}

#[test]
fn bitwise_truth_accepts_four_atoms_but_rejects_five() {
    let mut a = Arena::new();
    let atoms = std::array::from_fn::<_, 5, _>(|_| a.opaque("independent", Width::W64));
    let mut union = atoms[0];
    for &atom in &atoms[1..4] {
        union = raw_bin(&mut a, BinOp::Or, union, atom, Width::W64);
    }
    assert_eq!(
        bitwise_truth::simplify(&mut a, BinOp::And, union, atoms[0], Width::W64),
        Some(atoms[0])
    );
    union = raw_bin(&mut a, BinOp::Or, union, atoms[4], Width::W64);
    assert_eq!(
        bitwise_truth::simplify(&mut a, BinOp::And, union, atoms[0], Width::W64),
        None
    );
}

#[test]
fn bitwise_truth_shared_dag_is_memoized_and_node_and_visit_budgets_are_bounded() {
    let mut a = Arena::new();
    let x = a.opaque("x", Width::W64);
    let mut shared = x;
    // 31 existing unique nodes, 62 calls including memo hits. Recursive
    // expansion without memoization would take over two billion visits.
    for _ in 0..30 {
        shared = raw_bin(&mut a, BinOp::Xor, shared, shared, Width::W64);
    }
    assert_eq!(
        bitwise_truth::simplify(&mut a, BinOp::Or, shared, x, Width::W64),
        Some(x)
    );
    shared = raw_bin(&mut a, BinOp::Xor, shared, shared, Width::W64);
    assert_eq!(
        bitwise_truth::simplify(&mut a, BinOp::Or, shared, x, Width::W64),
        None
    );
    // A long path hits the independent invocation budget before unwinding;
    // it must fail closed rather than continue or overflow the host stack.
    for _ in 0..256 {
        shared = raw_not(&mut a, shared, Width::W64);
    }
    assert_eq!(
        bitwise_truth::simplify(&mut a, BinOp::Or, shared, x, Width::W64),
        None
    );
}

#[test]
fn bitwise_truth_nonuniform_masks_do_not_act_like_boolean_scalars() {
    let mut a = Arena::new();
    let x = a.opaque("x", Width::W64);
    let low = a.constant(1, Width::W64);
    let high = a.constant(0x8000_0000_0000_0000, Width::W64);
    let left = raw_bin(&mut a, BinOp::And, x, low, Width::W64);
    let right = raw_bin(&mut a, BinOp::And, x, high, Width::W64);
    // Treating both nonzero constants as true would falsely turn this into 0.
    assert_eq!(
        bitwise_truth::simplify(&mut a, BinOp::Xor, left, right, Width::W64),
        None
    );
    let raw = raw_bin(&mut a, BinOp::Xor, left, right, Width::W64);
    assert_eq!(evaluate(&a, raw, &HashMap::from([(x, 1)])), 1);
    assert_eq!(evaluate(&a, raw, &HashMap::from([(x, 1 << 63)])), 1 << 63);
}

#[test]
fn bitwise_truth_observed_decimal_missing_canonical_subexpressions() {
    // Fresh normal-mode 1811C7FF0 diagnostic after descendant-only proof:
    // Y | ((Q ^ Y) ^ (Y & ~Q)), plus ~((Y & ~Q) ^ ~Y), with Y=4Q.
    // Their canonicals (Q|Y) and (Q&Y) need not already be descendants.
    let mut a = Arena::new();
    let x = a.init_reg(Reg::Rdx);
    let ten = a.constant(10, Width::W64);
    let q = a.bin(BinOp::UDiv, x, ten);
    let four = a.constant(4, Width::W64);
    let y = a.bin(BinOp::Mul, q, four);
    let nq = a.not(q);
    let ny = a.not(y);
    let residual = a.bin(BinOp::And, y, nq);
    let difference = a.bin(BinOp::Xor, q, y);
    let nested = a.bin(BinOp::Xor, difference, residual);
    let left = a.bin(BinOp::Or, y, nested);
    let hidden_common = a.bin(BinOp::Xor, residual, ny);
    let right = a.not(hidden_common);
    let q5 = a.bin(BinOp::Add, left, right);
    let two = a.constant(2, Width::W64);
    let q10 = a.bin(BinOp::Mul, q5, two);
    let remainder = a.bin(BinOp::Sub, x, q10);
    let expected = a.bin(BinOp::URem, x, ten);
    assert_eq!(remainder, expected, "{}", render(&a, remainder, 20));
}

#[test]
fn bitwise_truth_missing_canonical_complement_can_be_created() {
    let mut a = Arena::new();
    let x = a.opaque("x", Width::W64);
    let y = a.opaque("y", Width::W64);
    let union = raw_bin(&mut a, BinOp::Or, x, y, Width::W64);
    let xor = raw_bin(&mut a, BinOp::Xor, x, y, Width::W64);
    let common = raw_bin(&mut a, BinOp::Xor, union, xor, Width::W64);
    let complement = raw_not(&mut a, common, Width::W64);
    let result = a.bin(BinOp::Or, complement, xor);
    // (not(X&Y)) | (X^Y) == not(X&Y); that canonical AND is
    // deliberately absent from the input DAG.
    let expected_common = raw_bin(&mut a, BinOp::And, x, y, Width::W64);
    let expected = raw_not(&mut a, expected_common, Width::W64);
    assert_eq!(result, expected, "{}", render(&a, result, 20));
}

// Build each Boolean function as a redundant sum of minterms. Neither this
// recipe nor evaluate() uses the production truth table/canonical mapping.
fn synthesis_fixtures(a: &mut Arena, x: Ref, y: Ref, width: Width) -> Vec<(Ref, Ref)> {
    let nx = raw_not(a, x, width);
    let ny = raw_not(a, y, width);
    let terms = [
        raw_bin(a, BinOp::And, nx, ny, width),
        raw_bin(a, BinOp::And, x, ny, width),
        raw_bin(a, BinOp::And, nx, y, width),
        raw_bin(a, BinOp::And, x, y, width),
    ];
    let zero = a.constant(0, width);
    let mut fixtures = Vec::new();
    for function in 0..16 {
        let mut source = zero;
        for (i, &term) in terms.iter().enumerate() {
            if function & (1 << i) != 0 {
                source = raw_bin(a, BinOp::Or, source, term, width);
            }
        }
        // A raw redundant root exercises all 16 functions, including NOTs,
        // constants and positive atoms, without existing peephole shortcuts.
        let original = raw_bin(a, BinOp::Or, source, source, width);
        let simplified =
            bitwise_truth::simplify(a, BinOp::Or, source, source, width).unwrap_or(source);
        assert_ne!(original, simplified);
        fixtures.push((original, simplified));
    }
    // The actual 1811C7FF0 identities require creation of absent subexpressions.
    let difference = raw_bin(a, BinOp::Xor, x, y, width);
    let residual = raw_bin(a, BinOp::And, y, nx, width);
    for (op, left, right) in [
        (BinOp::Xor, difference, residual),
        (BinOp::Xor, residual, ny),
    ] {
        let original = raw_bin(a, op, left, right, width);
        let simplified = bitwise_truth::simplify(a, op, left, right, width).unwrap();
        assert_ne!(original, simplified);
        fixtures.push((original, simplified));
    }
    fixtures
}

#[test]
fn bitwise_truth_synthesis_matches_every_byte_pair() {
    let mut a = Arena::new();
    let x = a.opaque("x", Width::W8);
    let y = a.opaque("y", Width::W8);
    let fixtures = synthesis_fixtures(&mut a, x, y, Width::W8);
    for vx in 0..=255 {
        for vy in 0..=255 {
            let inputs = HashMap::from([(x, vx), (y, vy)]);
            for &(original, simplified) in &fixtures {
                assert_eq!(
                    evaluate(&a, original, &inputs),
                    evaluate(&a, simplified, &inputs)
                );
            }
        }
    }
}

#[test]
fn bitwise_truth_synthesis_matches_wide_random_and_boundary_values() {
    let mut seed = 0xb735_f490_21da_8081u64;
    for width in [Width::W16, Width::W32, Width::W64] {
        let mut a = Arena::new();
        let x = a.opaque("x", width);
        let y = a.opaque("y", width);
        let fixtures = synthesis_fixtures(&mut a, x, y, width);
        let mut values = vec![0, 1, 2, width.mask(), width.mask() >> 1, width.sign_bit()];
        values.extend([
            0xaaaa_aaaa_aaaa_aaaa,
            0x5555_5555_5555_5555,
            0x8000_0000_0000_0001,
        ]);
        for _ in 0..4096 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            values.push(seed);
        }
        for (i, &vx) in values.iter().enumerate() {
            let vy = values[(i * 29 + 11) % values.len()];
            let inputs = HashMap::from([(x, vx), (y, vy)]);
            for &(original, simplified) in &fixtures {
                assert_eq!(
                    evaluate(&a, original, &inputs),
                    evaluate(&a, simplified, &inputs)
                );
            }
        }
    }
}

#[test]
fn bitwise_truth_synthesis_rejects_non_decreasing_full_dag_and_closure_budget() {
    let mut a = Arena::new();
    let x = a.opaque("x", Width::W64);
    let nx = raw_not(&mut a, x, Width::W64);
    let nnx = raw_not(&mut a, nx, Width::W64);
    let factor = a.constant(17, Width::W64);
    let y = raw_bin(&mut a, BinOp::Mul, nnx, factor, Width::W64);
    // Local Boolean cost drops from 5 to 3; full DAG cost is unchanged because
    // the opaque Mul retains both NOTs. Reject without interning any new node.
    let before = a.nodes.len();
    assert_eq!(
        bitwise_truth::simplify(&mut a, BinOp::And, nnx, y, Width::W64),
        None
    );
    assert_eq!(a.nodes.len(), before);

    let mut large = x;
    for _ in 0..300 {
        large = raw_bin(&mut a, BinOp::Mul, large, factor, Width::W64);
    }
    let negated = raw_not(&mut a, large, Width::W64);
    let restored = raw_not(&mut a, negated, Width::W64);
    let before = a.nodes.len();
    assert_eq!(
        bitwise_truth::simplify(&mut a, BinOp::And, restored, y, Width::W64),
        None
    );
    assert_eq!(a.nodes.len(), before);
}

#[test]
fn bitwise_truth_synthesis_never_treats_per_lane_functions_as_uniform() {
    let mut a = Arena::new();
    let x = a.opaque("x", Width::W8);
    let y = a.opaque("y", Width::W8);
    let mask = a.constant(0b0101_1010, Width::W8);
    let xor = raw_bin(&mut a, BinOp::Xor, x, y, Width::W8);
    let nxor = raw_not(&mut a, xor, Width::W8);
    let original = raw_bin(&mut a, BinOp::Xor, nxor, mask, Width::W8);
    let before = a.nodes.len();
    assert_eq!(
        bitwise_truth::simplify(&mut a, BinOp::Xor, nxor, mask, Width::W8),
        None
    );
    assert_eq!(a.nodes.len(), before);
    for vx in 0..=255 {
        for vy in 0..=255 {
            assert_eq!(
                evaluate(&a, original, &HashMap::from([(x, vx), (y, vy)])),
                (!(vx ^ vy) ^ 0x5a) & 255
            );
        }
    }
}
