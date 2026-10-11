//! Observe exception metadata at every interpreted source instruction.
//! Cloned/forked evaluators share one audit so discarded paths cannot erase risk.
use super::exception::{Coverage, CoverageKind, ExceptionTable};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc, sync::Arc};

#[derive(Clone)]
pub struct SourceProtection {
    table: Arc<ExceptionTable>,
    state: Rc<RefCell<Observation>>,
    reject: bool,
}

#[cfg(test)]
#[path = "protection_tests.rs"]
mod tests;

#[derive(Default)]
struct Observation {
    clean_interval: Option<(u64, u64)>,
    risks: BTreeMap<u64, Coverage>,
}

impl SourceProtection {
    pub fn new(table: Arc<ExceptionTable>, reject: bool) -> Self {
        Self {
            table,
            state: Rc::new(RefCell::new(Observation::default())),
            reject,
        }
    }

    /// Return false only when preserve mode must stop interpretation.
    pub fn observe(&self, address: u64, size: usize) -> bool {
        let end = address.checked_add(size as u64).unwrap_or(address);
        let mut state = self.state.borrow_mut();
        if state
            .clean_interval
            .is_some_and(|(lo, hi)| lo <= address && address < end && end <= hi)
        {
            return true;
        }
        let coverage = self.table.coverage_range_va(address, end);
        if coverage.kind == CoverageKind::Unprotected {
            // Cache only an actual fully inspected runtime interval. Gaps are
            // queried again rather than accidentally caching a neighbouring EH range.
            if let Some(rva) = address
                .checked_sub(self.table.image_base)
                .and_then(|r| u32::try_from(r).ok())
            {
                let index = self.table.records.partition_point(|r| r.end_rva <= rva);
                if let Some(record) = self.table.records.get(index) {
                    if record.begin_rva <= rva && record.coverage.kind == CoverageKind::Unprotected
                    {
                        state.clean_interval = Some((
                            self.table.image_base + record.begin_rva as u64,
                            self.table.image_base + record.end_rva as u64,
                        ));
                    }
                }
            }
            true
        } else {
            // One diagnostic per containing metadata record; unknown gaps retain
            // their exact PC. Evidence is never replaced by a later clean path.
            let key = coverage
                .runtime_ranges
                .first()
                .map(|r| self.table.image_base + r.0 as u64)
                .unwrap_or(address);
            state
                .risks
                .entry(key)
                .and_modify(|previous| {
                    // A wider instruction/summary at the same record can expose
                    // another handler or Unknown after an earlier Handler hit.
                    // Deduplication must not erase that later evidence.
                    if coverage.kind == CoverageKind::Unknown {
                        previous.kind = CoverageKind::Unknown;
                    }
                    for handler in &coverage.handlers {
                        if !previous.handlers.contains(handler) {
                            previous.handlers.push(handler.clone());
                        }
                    }
                    for issue in &coverage.issues {
                        if !previous.issues.contains(issue) {
                            previous.issues.push(issue.clone());
                        }
                    }
                    for range in &coverage.runtime_ranges {
                        if !previous.runtime_ranges.contains(range) {
                            previous.runtime_ranges.push(*range);
                        }
                    }
                })
                .or_insert(coverage);
            !self.reject
        }
    }

    pub fn risks(&self) -> Vec<(u64, Coverage)> {
        self.state
            .borrow()
            .risks
            .iter()
            .map(|(pc, c)| (*pc, c.clone()))
            .collect()
    }
}
