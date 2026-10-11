use super::*;

const BASE: u64 = 0x180000000;

fn w16(data: &mut [u8], at: usize, value: u16) {
    data[at..at + 2].copy_from_slice(&value.to_le_bytes());
}
fn w32(data: &mut [u8], at: usize, value: u32) {
    data[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

/// A complete PE32+ image with distinct executable .text and metadata .rdata.
/// Deliberately go through the real PeFile parser and raw RVA mapping.
fn fixture(records: &[(u32, u32, u32)]) -> Vec<u8> {
    let mut data = vec![0u8; 0x3400];
    data[0..2].copy_from_slice(b"MZ");
    w32(&mut data, 0x3c, 0x80);
    data[0x80..0x84].copy_from_slice(b"PE\0\0");
    w16(&mut data, 0x84, 0x8664);
    w16(&mut data, 0x86, 2);
    w16(&mut data, 0x94, 0xf0);
    let opt = 0x98;
    w16(&mut data, opt, 0x20b);
    data[opt + 24..opt + 32].copy_from_slice(&BASE.to_le_bytes());
    w32(&mut data, opt + 32, 0x1000);
    w32(&mut data, opt + 36, 0x200);
    w32(&mut data, opt + 56, 0x5000);
    w32(&mut data, opt + 60, 0x400);
    w32(&mut data, opt + 108, 16);
    if !records.is_empty() {
        w32(&mut data, opt + 112 + 24, 0x3000);
        w32(&mut data, opt + 112 + 28, records.len() as u32 * 12);
    }
    let section = opt + 0xf0;
    data[section..section + 5].copy_from_slice(b".text");
    w32(&mut data, section + 8, 0x2000);
    w32(&mut data, section + 12, 0x1000);
    w32(&mut data, section + 16, 0x2000);
    w32(&mut data, section + 20, 0x400);
    w32(&mut data, section + 36, 0x60000020);
    let section = section + 40;
    data[section..section + 6].copy_from_slice(b".rdata");
    w32(&mut data, section + 8, 0x2000);
    w32(&mut data, section + 12, 0x3000);
    w32(&mut data, section + 16, 0x1000);
    w32(&mut data, section + 20, 0x2400);
    w32(&mut data, section + 36, 0x40000040);
    data[0x400..0x2400].fill(0xc3);
    for (index, &(begin, end, unwind)) in records.iter().enumerate() {
        record(&mut data, 0x3000 + index as u32 * 12, (begin, end, unwind));
    }
    data
}

fn raw(rva: u32) -> usize {
    (rva - 0x1000 + 0x400) as usize
}
fn put(data: &mut [u8], rva: u32, bytes: &[u8]) {
    let at = raw(rva);
    data[at..at + bytes.len()].copy_from_slice(bytes);
}
fn record(data: &mut [u8], rva: u32, values: (u32, u32, u32)) {
    for (index, value) in [values.0, values.1, values.2].iter().enumerate() {
        w32(data, raw(rva) + index * 4, *value);
    }
}
fn table(data: Vec<u8>) -> ExceptionTable {
    ExceptionTable::parse(&PeFile::parse(data).unwrap())
}
fn plain(data: &mut [u8], rva: u32) {
    put(data, rva, &[1, 0, 0, 0]);
}
fn handler(data: &mut [u8], rva: u32) {
    put(data, rva, &[0x19, 4, 1, 0, 4, 0x32, 0xab, 0xcd]);
    w32(data, raw(rva) + 8, 0x2000);
}

#[test]
fn exact_interior_and_half_open_boundaries_find_handler() {
    let mut data = fixture(&[(0x1100, 0x1200, 0x3100)]);
    handler(&mut data, 0x3100);
    let parsed = table(data);
    assert!(parsed.global_errors.is_empty());
    for rva in [0x1100, 0x1101, 0x11ff] {
        assert_eq!(parsed.coverage_rva(rva).kind, CoverageKind::Handler);
    }
    assert_eq!(parsed.coverage_rva(0x1200).kind, CoverageKind::Unprotected);
    assert_eq!(
        parsed.coverage_range_va(BASE + 0x1080, BASE + 0x1101).kind,
        CoverageKind::Handler
    );
    assert_eq!(
        parsed.coverage_va(BASE + 0x1190).handlers[0].language_data_rva,
        0x310c
    );
    assert_eq!(parsed.coverage_va(BASE + 0x1190).handlers[0].flags, 3);
}

#[test]
fn odd_slot_padding_is_skipped_not_required_to_be_zero() {
    let mut data = fixture(&[(0x1100, 0x1200, 0x3100)]);
    handler(&mut data, 0x3100);
    assert_eq!(
        table(data).coverage_rva(0x1100).handlers[0].handler_rva,
        0x2000
    );
}

#[test]
fn chaininfo_reaches_handler_and_records_provenance() {
    let mut data = fixture(&[(0x1100, 0x1200, 0x3100)]);
    put(&mut data, 0x3100, &[0x21, 0, 0, 0]);
    record(&mut data, 0x3104, (0x1300, 0x1400, 0x3140));
    handler(&mut data, 0x3140);
    let coverage = table(data).coverage_rva(0x1180);
    assert_eq!(coverage.kind, CoverageKind::Handler);
    assert_eq!(coverage.handlers[0].unwind_chain, vec![0x3100, 0x3140]);
    assert_eq!(coverage.handlers[0].runtime_begin_rva, 0x1100);
}

#[test]
fn low_bit_indirect_runtime_encoding_is_not_an_unwind_header() {
    let mut data = fixture(&[(0x1100, 0x1200, 0x3181)]);
    record(&mut data, 0x3180, (0x1300, 0x1400, 0x3100));
    handler(&mut data, 0x3100);
    let coverage = table(data).coverage_rva(0x1180);
    assert_eq!(coverage.kind, CoverageKind::Handler);
    assert_eq!(coverage.handlers[0].indirect_records, vec![0x3180]);
}

#[test]
fn indirect_then_chain_then_indirect_preserves_handler_relationship() {
    let mut data = fixture(&[(0x1100, 0x1200, 0x3181)]);
    record(&mut data, 0x3180, (0x1300, 0x1400, 0x3100));
    put(&mut data, 0x3100, &[0x21, 0, 0, 0]);
    record(&mut data, 0x3104, (0x1500, 0x1600, 0x31c1));
    record(&mut data, 0x31c0, (0x1700, 0x1800, 0x3140));
    handler(&mut data, 0x3140);
    let coverage = table(data).coverage_rva(0x1180);
    assert_eq!(coverage.kind, CoverageKind::Handler);
    assert_eq!(coverage.handlers[0].indirect_records, vec![0x3180, 0x31c0]);
    assert_eq!(coverage.handlers[0].unwind_chain, vec![0x3100, 0x3140]);
}

#[test]
fn chain_and_indirect_cycles_are_unknown() {
    let mut data = fixture(&[(0x1100, 0x1200, 0x3100)]);
    put(&mut data, 0x3100, &[0x21, 0, 0, 0]);
    record(&mut data, 0x3104, (0x1100, 0x1200, 0x3100));
    assert!(
        table(data)
            .coverage_rva(0x1180)
            .issues
            .join(" ")
            .contains("cycle")
    );
    let mut data = fixture(&[(0x1100, 0x1200, 0x3181)]);
    record(&mut data, 0x3180, (0x1300, 0x1400, 0x3181));
    assert!(
        table(data)
            .coverage_rva(0x1180)
            .issues
            .join(" ")
            .contains("cycle")
    );
}

#[test]
fn malformed_unwind_is_local_and_unknown_dominates_multi_range_query() {
    let mut data = fixture(&[
        (0x1100, 0x1200, 0x3100),
        (0x1300, 0x1400, 0x3120),
        (0x1500, 0x1600, 0x3140),
    ]);
    handler(&mut data, 0x3100);
    put(&mut data, 0x3120, &[2, 0, 0, 0]);
    plain(&mut data, 0x3140);
    let parsed = table(data);
    assert!(parsed.global_errors.is_empty());
    assert_eq!(parsed.coverage_rva(0x1501).kind, CoverageKind::Unprotected);
    let coverage = parsed.coverage_range_rva(0x1180, 0x1580);
    assert_eq!(coverage.kind, CoverageKind::Unknown);
    assert_eq!(coverage.handlers.len(), 1);
    assert_eq!(coverage.runtime_ranges.len(), 3);
}

#[test]
fn invalid_versions_flags_operations_and_frame_fields_fail_closed() {
    for bytes in [
        vec![0, 0, 0, 0],
        vec![2, 0, 0, 0],
        vec![3, 0, 0, 0],
        vec![0x41, 0, 0, 0],
        vec![0x29, 0, 0, 0],
        vec![1, 0, 0, 0x10],
        vec![1, 0, 0, 1],
        vec![1, 0, 0, 5],
        vec![1, 4, 1, 0, 4, 0x01, 0, 0],
        vec![1, 4, 1, 0, 4, 0x06, 0, 0],
        vec![1, 4, 1, 0, 5, 0x32, 0, 0],
        vec![1, 4, 2, 0, 2, 0x32, 4, 0x32],
        vec![1, 4, 2, 0, 4, 0x01, 0, 0],
        vec![1, 4, 1, 0, 4, 0x40, 0, 0],
    ] {
        let mut data = fixture(&[(0x1100, 0x1200, 0x3100)]);
        put(&mut data, 0x3100, &bytes);
        assert_eq!(
            table(data).coverage_rva(0x1180).kind,
            CoverageKind::Unknown,
            "{bytes:02x?}"
        );
    }
}

#[test]
fn valid_multi_slot_codes_do_not_treat_operands_as_code_offsets() {
    let mut data = fixture(&[(0x1100, 0x1200, 0x3100)]);
    // SAVE_NONVOL_FAR RBX at offset0x10008, SAVE_XMM128 XMM15 at0x1ff0,
    // ALLOC_LARGE0x208, SET_FPREG RBP, PUSH_NONVOL RBP.
    put(
        &mut data,
        0x3100,
        &[
            1, 24, 9, 0x15, 24, 0x35, 8, 0, 1, 0, 18, 0xf8, 0xff, 1, 10, 0x01, 0x41, 0, 5, 0x03, 1,
            0x50, 0xde, 0xad,
        ],
    );
    assert_eq!(
        table(data).coverage_rva(0x1180).kind,
        CoverageKind::Unprotected
    );
}

#[test]
fn chain_to_plain_and_separate_termination_handler_classify_distinctly() {
    let mut data = fixture(&[(0x1100, 0x1200, 0x3100), (0x1500, 0x1600, 0x3180)]);
    put(&mut data, 0x3100, &[0x21, 0, 0, 0]);
    record(&mut data, 0x3104, (0x1300, 0x1400, 0x3140));
    plain(&mut data, 0x3140);
    put(&mut data, 0x3180, &[0x11, 0, 0, 0]);
    w32(&mut data, raw(0x3184), 0x2000);
    let parsed = table(data);
    assert_eq!(parsed.coverage_rva(0x1101).kind, CoverageKind::Unprotected);
    let coverage = parsed.coverage_rva(0x1501);
    assert_eq!(coverage.kind, CoverageKind::Handler);
    assert_eq!(coverage.handlers[0].flags, 2);
}

#[test]
fn partial_tail_and_invalid_indirect_targets_remain_unknown() {
    for unwind in [0x3ff8, 0x3ffc, 0x3003, 0x4001, 0xfffffffdu32] {
        let mut data = fixture(&[(0x1100, 0x1200, unwind)]);
        if unwind == 0x3ff8 {
            put(&mut data, unwind, &[0x21, 0, 0, 0]);
        }
        if unwind == 0x3ffc {
            put(&mut data, unwind, &[0x09, 0, 0, 0]);
        }
        assert_eq!(table(data).coverage_rva(0x1180).kind, CoverageKind::Unknown);
    }
}

#[test]
fn absent_directory_is_based_on_declared_directory_count() {
    let mut data = fixture(&[(0x1100, 0x1200, 0x3100)]);
    // Bytes physically present beyond the declared array are not directories.
    w32(&mut data, 0x98 + 108, 3);
    let parsed = table(data);
    assert!(parsed.records.is_empty());
    assert!(parsed.global_errors.is_empty());
    assert_eq!(parsed.coverage_rva(0x1101).kind, CoverageKind::Unprotected);
}

#[test]
fn patch_overlap_protects_directory_and_exact_unwind_bytes() {
    let mut data = fixture(&[(0x1100, 0x1200, 0x1300)]);
    put(&mut data, 0x1300, &[1, 4, 1, 0, 4, 0x32, 0xcc, 0xcc]);
    let parsed = table(data);
    assert!(
        parsed
            .patch_metadata_overlap(BASE + 0x100, BASE + 0x105)
            .is_none()
    );
    assert!(
        parsed
            .patch_metadata_overlap(BASE + 0x3004, BASE + 0x3009)
            .is_some()
    );
    assert!(
        parsed
            .patch_metadata_overlap(BASE + 0x1306, BASE + 0x1308)
            .is_some()
    );
    assert!(
        parsed
            .patch_metadata_overlap(BASE + 0x1308, BASE + 0x130d)
            .is_none()
    );
    assert!(
        parsed
            .patch_metadata_overlap(BASE + 0x1100, BASE + 0x1105)
            .is_none()
    );
}

#[test]
fn only_referenced_header_bytes_are_exception_metadata() {
    let mut data = fixture(&[(0x1100, 0x1200, 0x300)]);
    data[0x300..0x304].copy_from_slice(&[1, 0, 0, 0]);
    let parsed = table(data);
    assert!(
        parsed
            .patch_metadata_overlap(BASE + 0x300, BASE + 0x304)
            .is_some()
    );
    assert!(
        parsed
            .patch_metadata_overlap(BASE + 0x304, BASE + 0x309)
            .is_none()
    );
    assert!(
        parsed
            .patch_metadata_overlap(BASE + 0x100, BASE + 0x105)
            .is_none()
    );
}

#[test]
fn unknown_unwind_in_executable_padding_reserves_container_tail() {
    let mut data = fixture(&[(0x1100, 0x1200, 0x1300)]);
    put(&mut data, 0x1300, &[0xcc, 0xcc, 0xcc, 0xcc]);
    let parsed = table(data);
    assert_eq!(parsed.coverage_rva(0x1500).kind, CoverageKind::Unprotected);
    assert_eq!(
        parsed.patch_metadata_overlap(BASE + 0x1500, BASE + 0x1505),
        // The adjacent original exception table [3000,300c) merges into it.
        Some((BASE + 0x1300, BASE + 0x300c))
    );
    assert!(
        parsed
            .patch_metadata_overlap(BASE + 0x12fc, BASE + 0x1300)
            .is_none()
    );
}

#[test]
fn language_specific_data_is_opaque_until_original_container_end() {
    let mut data = fixture(&[(0x1100, 0x1200, 0x1300)]);
    handler(&mut data, 0x1300);
    let parsed = table(data);
    assert!(
        parsed
            .patch_metadata_overlap(BASE + 0x2500, BASE + 0x2505)
            .is_some()
    );
    assert!(
        parsed
            .patch_metadata_overlap(BASE + 0x2ffc, BASE + 0x3000)
            .is_some()
    );
    assert!(
        parsed
            .patch_metadata_overlap(BASE + 0x3010, BASE + 0x3015)
            .is_none()
    );
}

#[test]
fn indirect_and_chain_metadata_in_code_ranges_remain_protected() {
    let mut data = fixture(&[(0x1100, 0x1200, 0x1381)]);
    record(&mut data, 0x1380, (0x1500, 0x1600, 0x1400));
    put(&mut data, 0x1400, &[0x21, 0, 0, 0]);
    record(&mut data, 0x1404, (0x1700, 0x1800, 0x1480));
    plain(&mut data, 0x1480);
    let parsed = table(data);
    for rva in [0x1380, 0x1388, 0x1400, 0x140c, 0x1480] {
        assert!(
            parsed
                .patch_metadata_overlap(BASE + rva, BASE + rva + 1)
                .is_some()
        );
    }
    assert!(
        parsed
            .patch_metadata_overlap(BASE + 0x1484, BASE + 0x1489)
            .is_none()
    );
    assert_eq!(parsed.coverage_rva(0x1100).kind, CoverageKind::Unprotected);
}

#[test]
fn metadata_overlap_invalid_query_and_global_error_fail_closed() {
    let parsed = table(fixture(&[]));
    assert!(parsed.patch_metadata_overlap(BASE, BASE).is_some());
    assert!(parsed.patch_metadata_overlap(BASE - 1, BASE + 1).is_some());
    assert!(
        parsed
            .patch_metadata_overlap(BASE + 0x5000, BASE + 0x5001)
            .is_some()
    );
    let parsed = table(fixture(&[(0x1200, 0x1100, 0x3100)]));
    assert!(
        parsed
            .patch_metadata_overlap(BASE + 0x1500, BASE + 0x1505)
            .is_some()
    );
}

#[test]
fn zero_unaligned_unmapped_truncated_and_zero_filled_unwind_are_unknown() {
    for rva in [0, 0x3102, 0x3ffc, 0x4000, 0x6000, u32::MAX - 3] {
        let mut data = fixture(&[(0x1100, 0x1200, rva)]);
        if rva == 0x3ffc {
            put(&mut data, rva, &[1, 4, 1, 0]);
        }
        assert_eq!(
            table(data).coverage_rva(0x1180).kind,
            CoverageKind::Unknown,
            "{rva:x}"
        );
    }
}

#[test]
fn chain_frame_mismatch_and_invalid_handler_or_chain_code_are_unknown() {
    let mut data = fixture(&[(0x1100, 0x1200, 0x3100)]);
    put(&mut data, 0x3100, &[0x21, 0, 0, 5]);
    record(&mut data, 0x3104, (0x1300, 0x1400, 0x3140));
    plain(&mut data, 0x3140);
    assert_eq!(table(data).coverage_rva(0x1180).kind, CoverageKind::Unknown);
    for handler_rva in [0, 0x3100, 0x4500, 0x6000, u32::MAX] {
        let mut data = fixture(&[(0x1100, 0x1200, 0x3100)]);
        handler(&mut data, 0x3100);
        w32(&mut data, raw(0x3108), handler_rva);
        assert_eq!(table(data).coverage_rva(0x1180).kind, CoverageKind::Unknown);
    }
}

#[test]
fn runtime_sort_overlap_duplicate_partial_and_mapping_errors_are_global() {
    for records in [
        vec![(0x1200, 0x1300, 0x3100), (0x1100, 0x1200, 0x3100)],
        vec![(0x1100, 0x1300, 0x3100), (0x1200, 0x1400, 0x3100)],
        vec![(0x1100, 0x1200, 0x3100), (0x1100, 0x1200, 0x3100)],
        vec![(0x3100, 0x3200, 0x3100)],
        vec![(0, 0, 0)],
        vec![(0x2000, 0x1000, 0x3100)],
    ] {
        let parsed = table(fixture(&records));
        assert!(!parsed.global_errors.is_empty());
        assert_eq!(parsed.coverage_rva(0x2000).kind, CoverageKind::Unknown);
    }
    let mut data = fixture(&[(0x1100, 0x1200, 0x3100)]);
    w32(&mut data, 0x98 + 112 + 28, 13);
    assert!(!table(data).global_errors.is_empty());
}

#[test]
fn invalid_directory_bounds_and_ambiguous_section_mapping_are_global() {
    for (offset, value) in [
        (0x98 + 108, 17),
        (0x98 + 112 + 24, 0x3ff8),
        (0x98 + 112 + 24, 0),
        (0x98 + 0xf0 + 40 + 12, 0x2000),
        (0x98 + 0xf0 + 40 + 20, 0x1000),
    ] {
        let mut data = fixture(&[(0x1100, 0x1200, 0x3100)]);
        w32(&mut data, offset, value);
        assert!(!table(data).global_errors.is_empty(), "{offset:x}");
    }
}

#[test]
fn absent_directory_leaf_queries_and_unsafe_range_lookup_agree() {
    let parsed = table(fixture(&[]));
    assert!(parsed.global_errors.is_empty());
    assert_eq!(parsed.coverage_rva(0x2000).kind, CoverageKind::Unprotected);
    for rva in [0, 0xfff, 0x3000, 0x4000, 0x5000] {
        assert_eq!(parsed.coverage_rva(rva).kind, CoverageKind::Unknown);
    }
    assert_eq!(
        parsed.coverage_range_rva(0x1100, 0x1100).kind,
        CoverageKind::Unknown
    );
    let mut data = fixture(&[(0x1100, 0x1200, 0x3100), (0x1300, 0x1400, 0x3120)]);
    handler(&mut data, 0x3100);
    put(&mut data, 0x3120, &[2, 0, 0, 0]);
    let parsed = table(data);
    let unsafe_ranges = parsed.unsafe_ranges();
    for rva in 0..0x5000 {
        let contained = unsafe_ranges
            .iter()
            .any(|&(lo, hi, _)| lo <= BASE + rva && BASE + rva < hi);
        assert_eq!(
            contained,
            parsed.coverage_va(BASE + rva).kind != CoverageKind::Unprotected,
            "{rva:x}"
        );
    }
}

/// Optional read-only audit of a supplied PE and inventory; never loads its code.
#[test]
#[ignore = "requires TVM_EXCEPTION_AUDIT_INPUT, TVM_EXCEPTION_AUDIT_INVENTORY and TVM_EXCEPTION_AUDIT_OUTPUT"]
fn offline_sample_exception_audit() {
    let input = std::env::var("TVM_EXCEPTION_AUDIT_INPUT").unwrap();
    let inventory = std::env::var("TVM_EXCEPTION_AUDIT_INVENTORY").unwrap();
    let output = std::env::var("TVM_EXCEPTION_AUDIT_OUTPUT").unwrap();
    let parsed = ExceptionTable::parse(&PeFile::load(std::path::Path::new(&input)).unwrap());
    let inventory: serde_json::Value =
        serde_json::from_slice(&std::fs::read(inventory).unwrap()).unwrap();
    let mut counts = std::collections::BTreeMap::new();
    let mut unknown_records = Vec::new();
    for record in &parsed.records {
        *counts
            .entry(format!("{:?}", record.coverage.kind))
            .or_insert(0usize) += 1;
        if record.coverage.kind == CoverageKind::Unknown {
            unknown_records.push(serde_json::json!({"begin": format!("{:#x}", record.begin_rva), "unwind": format!("{:#x}", record.unwind_data), "issues": record.coverage.issues}));
        }
    }
    let mut entries = Vec::new();
    let mut entry_counts = std::collections::BTreeMap::new();
    for item in inventory["emitted_map"].as_array().unwrap() {
        let va = u64::from_str_radix(item["entry"].as_str().unwrap().trim_start_matches("0x"), 16)
            .unwrap();
        let coverage = parsed.coverage_va(va);
        *entry_counts
            .entry(format!("{:?}", coverage.kind))
            .or_insert(0usize) += 1;
        let exact = parsed
            .records
            .iter()
            .any(|r| parsed.image_base + u64::from(r.begin_rva) == va);
        let handlers: Vec<_> = coverage.handlers.iter().map(|h| serde_json::json!({"begin": h.runtime_begin_rva, "end": h.runtime_end_rva, "handler": h.handler_rva, "flags": h.flags, "unwind_chain": h.unwind_chain, "indirect_records": h.indirect_records})).collect();
        entries.push(serde_json::json!({"entry": item["entry"], "kind": format!("{:?}", coverage.kind), "exact_runtime_begin": exact, "runtime_ranges": coverage.runtime_ranges, "issues": coverage.issues, "handlers": handlers,
            "five_byte_patch_metadata_overlap": parsed.patch_metadata_overlap(va, va.checked_add(5).unwrap())}));
    }
    let overlap_count = entries
        .iter()
        .filter(|row| !row["five_byte_patch_metadata_overlap"].is_null())
        .count();
    let report = serde_json::json!({"input": input, "runtime_count": parsed.records.len(), "global_errors": parsed.global_errors, "runtime_class_counts": counts, "unknown_records": unknown_records, "entry_class_counts": entry_counts, "entries": entries,
        "metadata_ranges_rva": parsed.metadata_ranges, "entry_five_byte_metadata_overlap_count": overlap_count});
    std::fs::write(output, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    println!(
        "runtime={counts:?}, entries={entry_counts:?}, global_errors={:?}",
        parsed.global_errors
    );
}
