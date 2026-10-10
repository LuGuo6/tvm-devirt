use super::*;
use crate::ir::BlockId;

fn origin(handler: u64, vip: Option<u64>, event_index: usize) -> CallOrigin {
    CallOrigin { block: BlockId { handler, vip }, event_index, site: handler + 5 }
}

#[test]
fn call_origin_clone_and_compaction_preserve_live_definitions_only() {
    let mut source = Arena::new();
    let dead = source.opaque("call_ret", Width::W64);
    source.bind_call_origin(dead, origin(0x1000, None, 1));
    let value = source.opaque("call_ret", Width::W64);
    let expected = origin(0x2000, Some(0x3456), 9);
    source.bind_call_origin(value, expected);
    let one = source.constant(1, Width::W64);
    let root = source.bin(BinOp::Add, value, one);
    let mut copy = source.clone();
    assert_eq!(copy.call_origin(value), Some(expected));
    let remap = copy.compact(&[root]);
    assert!(!remap.contains_key(&dead));
    assert_eq!(copy.call_origin(remap[&value]), Some(expected));
    assert_eq!(copy.call_origins.len(), 1);
    assert_eq!(source.call_origins.len(), 2);
    copy.compact(&[]);
    assert!(copy.call_origins.is_empty());
}

#[test]
fn call_origin_grafting_preserves_source_without_merging_opaque_identity() {
    let mut source = Arena::new();
    let value = source.opaque("call_ret", Width::W64);
    let expected = origin(0x1000, Some(7), 2);
    source.bind_call_origin(value, expected);
    let mut target = Arena::new();
    let first = target.graft(&source, value, &mut HashMap::new());
    let second = target.graft(&source, value, &mut HashMap::new());
    assert_ne!(first, second);
    assert_eq!(target.call_origin(first), Some(expected));
    assert_eq!(target.call_origin(second), Some(expected));
    let mut memo = HashMap::new();
    let shared = target.graft(&source, value, &mut memo);
    assert_eq!(shared, target.graft(&source, value, &mut memo));
}

#[test]
fn call_origin_local_opaque_numbers_do_not_conflate_different_definitions() {
    let mut a = Arena::new();
    let mut b = Arena::new();
    let av = a.opaque("call_ret", Width::W64);
    let bv = b.opaque("call_ret", Width::W64);
    assert_eq!(a.op(av), b.op(bv));
    let ao = origin(0x1000, Some(1), 0);
    let bo = origin(0x1000, Some(2), 0);
    a.bind_call_origin(av, ao);
    b.bind_call_origin(bv, bo);
    let mut dst = Arena::new();
    let ar = dst.graft(&a, av, &mut HashMap::new());
    let br = dst.graft(&b, bv, &mut HashMap::new());
    assert_ne!(ar, br);
    assert_eq!(dst.call_origin(ar), Some(ao));
    assert_eq!(dst.call_origin(br), Some(bo));
}

#[test]
fn call_origin_does_not_attach_to_seeded_parameters_or_unknown_values() {
    let mut source = Arena::new();
    let value = source.opaque("call_ret", Width::W64);
    source.bind_call_origin(value, origin(0x1000, None, 0));
    let unknown = source.opaque("call_ret", Width::W64);
    let mut dst = Arena::new();
    let param = dst.param(BlockRef(7), Reg::Rbx);
    let mut memo = HashMap::from([(value, param)]);
    assert_eq!(dst.graft(&source, value, &mut memo), param);
    assert_eq!(dst.call_origin(param), None);
    let copied_unknown = dst.graft(&source, unknown, &mut HashMap::new());
    assert_eq!(dst.call_origin(copied_unknown), None);
}

#[test]
fn call_origin_conflicting_rebind_rejects_without_changing_existing_evidence() {
    let mut source = Arena::new();
    let value = source.opaque("call_ret", Width::W64);
    let expected = origin(0x1000, None, 0);
    source.bind_call_origin(value, expected);
    source.bind_call_origin(value, expected);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        source.bind_call_origin(value, origin(0x1000, None, 1));
    }));
    assert!(result.is_err());
    assert_eq!(source.call_origin(value), Some(expected));
}
