use super::*;
use crate::binary::pe::PeFile;
use crate::ir::lift::{Emulator, Event, Step, Stop};

const BASE: u64 = 0x180000000;
const STACK: u64 = 0x7ffffff00000;

fn w16(data: &mut [u8], at: usize, value: u16) {
    data[at..at + 2].copy_from_slice(&value.to_le_bytes());
}
fn w32(data: &mut [u8], at: usize, value: u32) {
    data[at..at + 4].copy_from_slice(&value.to_le_bytes());
}
fn raw(rva: u32) -> usize {
    (rva - 0x1000 + 0x400) as usize
}

fn fixture(records: &[(u32, u32, CoverageKind)], code: &[(u32, &[u8])]) -> PeFile {
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
    for (offset, value) in [
        (32, 0x1000),
        (36, 0x200),
        (56, 0x5000),
        (60, 0x400),
        (108, 16),
    ] {
        w32(&mut data, opt + offset, value);
    }
    if !records.is_empty() {
        w32(&mut data, opt + 136, 0x3000);
        w32(&mut data, opt + 140, records.len() as u32 * 12);
    }
    let section = opt + 0xf0;
    data[section..section + 5].copy_from_slice(b".tvm0");
    for (offset, value) in [
        (8, 0x2000),
        (12, 0x1000),
        (16, 0x2000),
        (20, 0x400),
        (36, 0x60000020),
    ] {
        w32(&mut data, section + offset, value);
    }
    let section = section + 40;
    data[section..section + 6].copy_from_slice(b".rdata");
    for (offset, value) in [
        (8, 0x2000),
        (12, 0x3000),
        (16, 0x1000),
        (20, 0x2400),
        (36, 0x40000040),
    ] {
        w32(&mut data, section + offset, value);
    }
    data[0x400..0x2400].fill(0xc3);
    for (index, &(begin, end, kind)) in records.iter().enumerate() {
        let at = raw(0x3000) + 12 * index;
        let unwind = 0x3100 + index as u32 * 16;
        w32(&mut data, at, begin);
        w32(&mut data, at + 4, end);
        w32(&mut data, at + 8, unwind);
        data[raw(unwind)] = match kind {
            CoverageKind::Unprotected => 1,
            CoverageKind::Handler => 9,
            CoverageKind::Unknown => 2,
        };
        if kind == CoverageKind::Handler {
            w32(&mut data, raw(unwind) + 4, 0x2000);
        }
    }
    for &(rva, bytes) in code {
        data[raw(rva)..raw(rva) + bytes.len()].copy_from_slice(bytes);
    }
    PeFile::parse(data).unwrap()
}

fn audit(pe: &PeFile, reject: bool) -> SourceProtection {
    let table = ExceptionTable::parse(pe);
    assert!(table.global_errors.is_empty(), "{:?}", table.global_errors);
    SourceProtection::new(Arc::new(table), reject)
}

#[test]
fn instruction_straddling_handler_rejects_even_after_clean_cache_hit() {
    let pe = fixture(
        &[
            (0x1100, 0x1110, CoverageKind::Unprotected),
            (0x1110, 0x1200, CoverageKind::Handler),
        ],
        &[],
    );
    let audit = audit(&pe, true);
    assert!(audit.observe(BASE + 0x1100, 2));
    assert!(audit.observe(BASE + 0x110a, 2));
    assert!(!audit.observe(BASE + 0x110f, 2));
    assert_eq!(audit.risks()[0].1.kind, CoverageKind::Handler);
}

#[test]
fn gap_cache_does_not_cross_handler_or_non_code() {
    let pe = fixture(&[(0x1110, 0x1120, CoverageKind::Handler)], &[]);
    let audit = audit(&pe, true);
    assert!(audit.observe(BASE + 0x1100, 2));
    assert!(audit.observe(BASE + 0x110e, 2));
    assert!(!audit.observe(BASE + 0x110f, 2));
    assert!(audit.observe(BASE + 0x1120, 2));
    assert!(audit.observe(BASE + 0x2fff, 1));
    assert!(!audit.observe(BASE + 0x2fff, 2));
}

#[test]
fn discarded_fork_keeps_unknown_risk_and_analysis_mode_continues() {
    let pe = fixture(
        &[
            (0x1100, 0x1200, CoverageKind::Handler),
            (0x1300, 0x1400, CoverageKind::Unknown),
        ],
        &[],
    );
    let audit = audit(&pe, false);
    {
        let discarded = audit.clone();
        assert!(discarded.observe(BASE + 0x1310, 3));
    }
    assert_eq!(audit.risks()[0].1.kind, CoverageKind::Unknown);
    assert!(audit.observe(BASE + 0x1100, 5));
    assert!(audit.observe(BASE + 0x1101, 2));
    assert_eq!(audit.risks().len(), 2);
    assert!(audit.observe(BASE + 0x1500, 2));
    assert_eq!(audit.risks().len(), 2);
}

#[test]
fn same_key_wider_range_preserves_new_unknown_and_handlers() {
    let pe = fixture(
        &[
            (0x1100, 0x1105, CoverageKind::Unprotected),
            (0x1105, 0x1109, CoverageKind::Handler),
            (0x1109, 0x1110, CoverageKind::Unknown),
        ],
        &[],
    );
    let audit = audit(&pe, false);
    assert!(audit.observe(BASE + 0x1103, 4));
    assert!(audit.observe(BASE + 0x1103, 9));
    let risks = audit.risks();
    assert_eq!(risks.len(), 1);
    assert_eq!(risks[0].1.kind, CoverageKind::Unknown);
    assert_eq!(risks[0].1.handlers.len(), 1);
    assert_eq!(risks[0].1.issues.len(), 1);
    assert_eq!(risks[0].1.runtime_ranges.len(), 3);
}

#[test]
fn empty_overflow_and_outside_queries_fail_closed() {
    let pe = fixture(&[], &[]);
    let audit = audit(&pe, true);
    assert!(!audit.observe(BASE + 0x1100, 0));
    assert!(!audit.observe(u64::MAX - 1, 10));
    assert!(!audit.observe(BASE - 1, 1));
    assert!(!audit.observe(BASE + 0x5000, 1));
}

const PROBE: &[u8] = &[
    0x9c, 0x51, 0x50, 0x48, 0x8d, 0x4c, 0x24, 0x28, 0x66, 0x81, 0xe1, 0, 0xf0, 0x48, 0x8d, 0x44,
    0x24, 0x28, 0x48, 0x2b, 0x44, 0x24, 0x20, 0x48, 0x8d, 0x89, 0, 0xf0, 0xff, 0xff, 0x84, 0x01,
    0x48, 0x39, 0xc1, 0x0f, 0x8d, 0xee, 0xff, 0xff, 0xff, 0x58, 0x59, 0x9d, 0xc2, 0x08, 0,
];

#[test]
fn summarized_stack_probe_observes_complete_helper_interval() {
    // The helper START has no runtime entry; only its interior is protected.
    for kind in [CoverageKind::Handler, CoverageKind::Unknown] {
        for reject in [false, true] {
            let pe = fixture(
                &[(0x1140, 0x114f, kind)],
                &[
                    (0x1100, &[0x52, 0xe8, 0x1a, 0, 0, 0, 0xc3]),
                    (0x1120, PROBE),
                ],
            );
            let audit = audit(&pe, reject);
            let mut emu = Emulator::new(&pe, STACK);
            emu.source_protection = Some(audit.clone());
            assert!(matches!(emu.step(BASE + 0x1100), Step::Next(_)));
            let step = emu.step(BASE + 0x1101);
            if reject {
                assert!(matches!(step, Step::Stopped(Stop::Unsupported { .. })));
                assert!(
                    !emu.events
                        .iter()
                        .any(|event| matches!(event, Event::StackProbe { .. }))
                );
            } else {
                assert!(matches!(step, Step::Next(ip) if ip == BASE + 0x1106));
                assert!(
                    emu.events
                        .iter()
                        .any(|event| matches!(event, Event::StackProbe { .. }))
                );
            }
            assert_eq!(audit.risks().len(), 1);
            assert_eq!(audit.risks()[0].1.kind, kind);
        }
    }
}

#[test]
fn vip_probe_and_recovery_forks_share_all_risks() {
    // TEST ECX,ECX; JZ handler; RET; ... handler RET. Symbolic branch is forked.
    let pe = fixture(
        &[(0x1120, 0x1130, CoverageKind::Handler)],
        &[(0x1100, &[0x85, 0xc9, 0x74, 0x1c, 0xc3]), (0x1120, &[0xc3])],
    );
    for reject in [false, true] {
        let audit = audit(&pe, reject);
        let mut explorer = crate::vm::explore::Explorer::new(&pe, STACK);
        explorer.source_protection = Some(audit.clone());
        let cfg = explorer.recover(BASE + 0x1100);
        assert!(!audit.risks().is_empty());
        assert!(
            audit
                .risks()
                .iter()
                .any(|(_, c)| c.kind == CoverageKind::Handler)
        );
        if reject {
            assert!(cfg.unresolved > 0);
        }
    }
}
