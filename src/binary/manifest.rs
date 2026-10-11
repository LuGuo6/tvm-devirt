//! Byte-bound generation report. A report is evidence of layout, not equivalence.
use super::pe::PeFile;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:X}", Sha256::digest(bytes))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Range {
    pub kind: String,
    pub state: String,
    pub start_rva: u32,
    pub size: u32,
    pub sha256: String,
    pub entry_va: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntryOutcome {
    pub entry_va: u64,
    pub status: String,
    pub reason: Option<String>,
    pub body_va: Option<u64>,
    pub body_size: Option<u32>,
    pub source_exception_risks: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub schema: String,
    pub version: u32,
    pub original_sha256: String,
    pub base_sha256: String,
    pub output_sha256: String,
    pub output_size: usize,
    pub image_base: u64,
    pub complete_history: bool,
    pub exception_policy: String,
    pub whole_dll_runtime_verified: bool,
    pub ranges: Vec<Range>,
    pub entries: Vec<EntryOutcome>,
    pub limits: Vec<String>,
}

impl Manifest {
    pub fn new(
        original: &PeFile,
        base: &PeFile,
        analysis_only: bool,
        parent: Option<&Self>,
    ) -> Self {
        Self {
            schema: "tvm-devirt-layout".into(), version: 2,
            original_sha256: sha256(&original.data), base_sha256: sha256(&base.data),
            output_sha256: String::new(), output_size: base.data.len(), image_base: base.image_base,
            complete_history: parent.map_or(original.data==base.data, |m| m.complete_history),
            exception_policy: if analysis_only { "analysis-only" } else { "preserve" }.into(),
            whole_dll_runtime_verified: false,
            ranges: parent.map(|p| p.ranges.clone()).unwrap_or_default(),
            entries: parent.map(|p| p.entries.clone()).unwrap_or_default(),
            limits: vec!["Source-PC exception screening preserves known/unknown handler-dependent entries; it does not translate language handlers or prove whole-program equivalence.".into(),
                "Original native/VM bytes may still need IDA analysis repair. Do not equate fewer warnings or emitted counts with runtime compatibility.".into()],
        }
    }

    /// Check generated evidence against the final bytes, including entry routing.
    /// This intentionally cannot authorize an incremental base or prove semantics.
    pub fn validate_output(&self, output: &PeFile) -> Result<()> {
        if self.schema != "tvm-devirt-layout"
            || self.version != 2
            || self.image_base != output.image_base
            || self.output_sha256 != sha256(&output.data)
            || self.output_size != output.data.len()
            || self.whole_dll_runtime_verified
            || !["preserve", "analysis-only"].contains(&self.exception_policy.as_str())
        {
            bail!("generation manifest does not match output or supported schema");
        }
        let table = super::exception::ExceptionTable::parse(output);
        if !table.global_errors.is_empty() {
            bail!(
                "generated exception directory is invalid: {:?}",
                table.global_errors
            );
        }
        for range in &self.ranges {
            range
                .start_rva
                .checked_add(range.size)
                .context("manifest range overflows")?;
            let address = output
                .image_base
                .checked_add(u64::from(range.start_rva))
                .context("manifest address overflows")?;
            let bytes = output
                .read_va(address, range.size as usize)
                .context("unmapped manifest range")?;
            if bytes.len() != range.size as usize || sha256(bytes) != range.sha256 {
                bail!("output range hash mismatch at {:#x}", range.start_rva);
            }
            if range.state == "active"
                && [
                    "code",
                    "code-section",
                    "unwind",
                    "runtime-table",
                    "metadata-section",
                ]
                .contains(&range.kind.as_str())
            {
                let section = output
                    .section_for_rva(range.start_rva)
                    .context("manifest range has no section")?;
                if ["code", "code-section"].contains(&range.kind.as_str()) {
                    if !section.is_executable() || section.is_writable() {
                        bail!("generated code is not RX");
                    }
                } else if section.is_executable() || section.is_writable() {
                    bail!("generated metadata is executable or writable");
                }
            }
        }
        let mut seen = std::collections::HashSet::new();
        for entry in &self.entries {
            if !seen.insert(entry.entry_va) {
                bail!("duplicate manifest entry");
            }
            if entry.status != "emitted" {
                continue;
            }
            if self.exception_policy == "preserve" && !entry.source_exception_risks.is_empty() {
                bail!("preserve-mode emitted entry retains source exception risks");
            }
            let body = entry.body_va.context("emitted entry has no body")?;
            let size = entry.body_size.context("emitted entry has no body size")?;
            let rva = output
                .va_to_rva(body)
                .context("emitted body outside image")?;
            if !self.ranges.iter().any(|r| {
                r.kind == "code"
                    && r.state == "active"
                    && r.start_rva == rva
                    && r.size == size
                    && r.entry_va == Some(entry.entry_va)
            }) {
                bail!("emitted entry lacks matching active code range");
            }
            let route = output
                .read_va(entry.entry_va, 2)
                .context("unmapped entry route")?;
            if route.len() != 2 {
                bail!("truncated entry route");
            }
            let jump = if route[0] == 0xeb {
                (entry.entry_va as i128 + 2 + i128::from(route[1] as i8))
                    .try_into()
                    .context("relay address overflows")?
            } else {
                entry.entry_va
            };
            let bytes = output.read_va(jump, 5).context("unmapped redirect")?;
            if bytes.len() != 5 || bytes[0] != 0xe9 {
                bail!("entry has no near redirect");
            }
            let displacement = i32::from_le_bytes(bytes[1..5].try_into()?);
            if jump as i128 + 5 + i128::from(displacement) != body as i128 {
                bail!("entry route does not reach manifest body");
            }
            let end = rva.checked_add(size).context("body range overflows")?;
            if !table
                .records
                .iter()
                .any(|r| r.begin_rva == rva && r.end_rva == end)
            {
                bail!("generated body lacks exact runtime-function record");
            }
        }
        Ok(())
    }

    pub fn record_range(
        &mut self,
        kind: &str,
        state: &str,
        rva: u32,
        bytes: &[u8],
        entry_va: Option<u64>,
    ) -> Result<()> {
        let size = u32::try_from(bytes.len()).context("manifest range exceeds PE32+ RVA extent")?;
        rva.checked_add(size).context("manifest range overflow")?;
        if size == 0 {
            return Ok(());
        }
        if let Some(previous) = self
            .ranges
            .iter()
            .find(|r| r.kind == kind && r.start_rva == rva && r.size == size)
        {
            if previous.sha256 != sha256(bytes)
                || previous.state != state
                || previous.entry_va != entry_va
            {
                bail!("conflicting manifest range at {rva:#x}");
            }
        } else {
            self.ranges.push(Range {
                kind: kind.into(),
                state: state.into(),
                start_rva: rva,
                size,
                sha256: sha256(bytes),
                entry_va,
            });
        }
        Ok(())
    }

    pub fn entry(&mut self, row: EntryOutcome) {
        if let Some(previous) = self.entries.iter().position(|r| r.entry_va == row.entry_va) {
            if row.status == "emitted" {
                for r in &mut self.ranges {
                    if r.kind == "code" && r.entry_va == Some(row.entry_va) {
                        r.state = "retired".into();
                    }
                }
                self.entries[previous] = row;
            }
            // A rejected new attempt does not undo an existing accepted redirect.
        } else {
            self.entries.push(row);
        }
    }

    pub fn finish(&mut self, output: &[u8]) {
        self.output_sha256 = sha256(output);
        self.output_size = output.len();
    }
}
