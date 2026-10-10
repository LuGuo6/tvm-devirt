mod binary;
mod cli;
mod codegen;
mod ir;
mod vm;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use cli::diag::cmd_unresolved;
use cli::emit::{cmd_devirt, cmd_write_devirt};
use cli::inspect::{cmd_dis, cmd_entries, cmd_info, cmd_trace, cmd_vmctx};
use cli::recover::{cmd_cfg, cmd_lower, cmd_ssa};

#[derive(Parser)]
#[command(
    name = "tvm-devirt",
    about = "static devirtualizer for Tencent VM (TVM)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Dump PE layout
    Info {
        input: PathBuf,
        #[arg(long, default_value = vm::DEFAULT_VM_SECTION)]
        vm_section: String,
    },
    /// List discovered virtualized function entry trampolines.
    Entries {
        input: PathBuf,
        #[arg(long, default_value = vm::DEFAULT_VM_SECTION)]
        vm_section: String,
        /// Print all entries instead of a summary.
        #[arg(long)]
        all: bool,
    },
    /// Linear disassembly at a virtual address
    Dis {
        input: PathBuf,
        #[arg(value_parser = parse_u64)]
        va: u64,
        #[arg(long, default_value_t = 60)]
        count: usize,
    },
    /// Guided symbolic evaluation of a virtualized function.
    Trace {
        input: PathBuf,
        /// Start VA: either the entry trampoline or a .tvm0 address.
        #[arg(value_parser = parse_u64)]
        va: u64,
        /// Instruction step budget.
        #[arg(long, default_value_t = 200_000)]
        steps: usize,
        /// Concrete value RSP is initialised to.
        #[arg(long, value_parser = parse_u64, default_value_t = vm::DEFAULT_STACK_BASE)]
        stack_base: u64,
        /// Print the guest events that were recorded.
        #[arg(long)]
        events: bool,
        /// Print the resolved VM handler transitions.
        #[arg(long)]
        dispatches: bool,
        /// Disassemble the last N executed instructions.
        #[arg(long, default_value_t = 0)]
        tail: usize,
        /// Stop at this VA and dump the register state (debugging aid).
        #[arg(long, value_parser = parse_u64)]
        break_at: Option<u64>,
    },
    /// Recover the control flow graph of a virtualized function.
    Cfg {
        input: PathBuf,
        /// Start VA: entry trampoline or a .tvm0 address.
        #[arg(value_parser = parse_u64)]
        va: u64,
        #[arg(long, value_parser = parse_u64, default_value_t = vm::DEFAULT_STACK_BASE)]
        stack_base: u64,
        /// Per-path instruction budget.
        #[arg(long, default_value_t = 400_000)]
        steps: usize,
        /// Maximum blocks to recover.
        #[arg(long, default_value_t = 512)]
        max_blocks: usize,
        /// Print each recovered block's guest events.
        #[arg(long)]
        events: bool,
        /// Print each block's exit register state.
        #[arg(long)]
        state: bool,
        /// Wall-clock limit in seconds for the whole recovery.
        #[arg(long, default_value_t = 120)]
        timeout: u64,
        /// Section holding VM bytecode.
        #[arg(long, default_value = vm::DEFAULT_VM_SECTION)]
        vm_section: String,
    },
    /// Dump the VM context window around RBP, to locate the guest register image.
    Vmctx {
        input: PathBuf,
        #[arg(value_parser = parse_u64)]
        va: u64,
        #[arg(long, value_parser = parse_u64, default_value_t = vm::DEFAULT_STACK_BASE)]
        stack_base: u64,
        /// Stop after this many VM instructions & dump.
        #[arg(long, default_value_t = 400_000)]
        steps: usize,
        /// Bytes of context to dump either side of RBP.
        #[arg(long, default_value_t = 0x120)]
        window: u64,
    },
    /// Show SSA parameters and phi nodes for a recovered function.
    Ssa {
        input: PathBuf,
        #[arg(value_parser = parse_u64)]
        va: u64,
        #[arg(long, value_parser = parse_u64, default_value_t = vm::DEFAULT_STACK_BASE)]
        stack_base: u64,
        #[arg(long, default_value_t = 400_000)]
        steps: usize,
        #[arg(long, default_value_t = 256)]
        max_blocks: usize,
        #[arg(long, default_value_t = 120)]
        timeout: u64,
        /// Only show phis with more than one distinct incoming value.
        #[arg(long)]
        joins_only: bool,
        /// Section holding VM bytecode.
        #[arg(long, default_value = vm::DEFAULT_VM_SECTION)]
        vm_section: String,
    },
    /// Schedule a recovered function into linear three-address code.
    Lower {
        input: PathBuf,
        #[arg(value_parser = parse_u64)]
        va: u64,
        #[arg(long, value_parser = parse_u64, default_value_t = vm::DEFAULT_STACK_BASE)]
        stack_base: u64,
        #[arg(long, default_value_t = 400_000)]
        steps: usize,
        #[arg(long, default_value_t = 256)]
        max_blocks: usize,
        #[arg(long, default_value_t = 120)]
        timeout: u64,
        /// Section holding VM bytecode.
        #[arg(long, default_value = vm::DEFAULT_VM_SECTION)]
        vm_section: String,
    },
    /// Devirtualize all functions and write a patched binary.
    WriteDevirt {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, default_value_t = 500_000)]
        steps: usize,
        /// Maximum blocks to recover per function.
        #[arg(long, default_value_t = 512)]
        max_blocks: usize,
        /// Wall-clock backstop per function, checked between recovered blocks.
        /// A single block can run past this value before the next check.
        #[arg(long, default_value_t = 600)]
        timeout: u64,
        /// Section holding VM bytecode.
        #[arg(long, default_value = vm::DEFAULT_VM_SECTION)]
        vm_section: String,
        /// Functions to recover concurrently. Defaults to `min(cores, 8)`.
        #[arg(long)]
        jobs: Option<usize>,
        /// Share of available logical cores for concurrency (1..=100), not a hard CPU limit.
        #[arg(long, value_parser = clap::value_parser!(u8).range(1..=100), conflicts_with = "jobs")]
        cpu_percent: Option<u8>,
        /// Recover only these discovered entry VAs (repeatable, comma-separated).
        #[arg(long, value_parser = parse_u64, value_delimiter = ',')]
        only_entry: Vec<u64>,
        /// Append to a compatible derived image while analyzing the original input.
        #[arg(long)]
        patch_base: Option<PathBuf>,
    },
    /// Devirtualize one function into x86 machine code.
    Devirt {
        input: PathBuf,
        #[arg(value_parser = parse_u64)]
        va: u64,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long, value_parser = parse_u64, default_value_t = vm::DEFAULT_STACK_BASE)]
        stack_base: u64,
        #[arg(long, default_value_t = 400_000)]
        steps: usize,
        /// Maximum blocks to recover. 512 is the historical default and is kept:
        /// raising it changes what the emitted function covers.
        #[arg(long, default_value_t = 512)]
        max_blocks: usize,
        #[arg(long, default_value_t = 120)]
        timeout: u64,
        /// Disassemble the emitted bytes after encoding.
        #[arg(long)]
        dis: bool,
    },
    /// Per-block diagnostics for a function whose recovery did not close.
    Unresolved {
        input: PathBuf,
        #[arg(value_parser = parse_u64)]
        va: u64,
        #[arg(long, value_parser = parse_u64, default_value_t = vm::DEFAULT_STACK_BASE)]
        stack_base: u64,
        /// Instruction step budget, set to match the run being diagnosed.
        #[arg(long, default_value_t = 400_000)]
        steps: usize,
        /// Maximum blocks to recover. The devirtualizer's own default is 512, and
        /// it is kept here so that diagnosing the baseline measures the baseline.
        /// Raising it is the point of an experiment, not of a comparison.
        #[arg(long, default_value_t = 512)]
        max_blocks: usize,
        /// Wall-clock limit in seconds.
        #[arg(long, default_value_t = 120)]
        timeout: u64,
        /// Write the JSON here instead of stdout.
        #[arg(long)]
        json: Option<PathBuf>,
    },
}

fn parse_u64(s: &str) -> Result<u64, String> {
    let t = s.trim();
    let r = t
        .strip_prefix("0x")
        .or_else(|| t.strip_prefix("0X"))
        .map_or_else(|| t.parse::<u64>(), |h| u64::from_str_radix(h, 16));
    r.map_err(|e| e.to_string())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let diagnostic_origin = std::env::var("TVM_DIAGNOSTIC_CONTEXT_ORIGIN").as_deref() == Ok("1");
    if std::env::var_os("TVM_DIAGNOSTIC_INLINE_CHKSTK").is_some() {
        anyhow::ensure!(!matches!(&cli.cmd, Cmd::WriteDevirt { .. } | Cmd::Devirt { .. }),
            "inline chkstk diagnostic is read-only; native emission is disabled");
    }
    let diagnostic_locations = std::env::var("TVM_DIAGNOSTIC_LOCATION_HOMES").as_deref() == Ok("1");
    anyhow::ensure!(!(diagnostic_origin && diagnostic_locations),
        "context-origin and location-home diagnostics must be run separately");
    if diagnostic_origin {
        eprintln!("DIAGNOSTIC ONLY: context-origin candidates lack a complete consumer proof; output is experimental and must not be counted as verified recovery or deployed.");
        if let Cmd::WriteDevirt { only_entry, .. } = &cli.cmd {
            anyhow::ensure!(only_entry.len() == 1,
                "diagnostic context-origin writeback requires exactly one --only-entry");
        }
    }
    if diagnostic_locations {
        eprintln!("DIAGNOSTIC ONLY: location-home recovery is experimental; output is excluded from verified recovery counts and deployment.");
        if let Cmd::WriteDevirt { only_entry, .. } = &cli.cmd {
            anyhow::ensure!(only_entry.len() == 1,
                "diagnostic location-home writeback requires exactly one --only-entry");
        }
    }
    match cli.cmd {
        Cmd::Info { input, vm_section } => cmd_info(&input, &vm_section),
        Cmd::Entries {
            input,
            vm_section,
            all,
        } => cmd_entries(&input, &vm_section, all),
        Cmd::Dis { input, va, count } => cmd_dis(&input, va, count),
        Cmd::Trace {
            input,
            va,
            steps,
            stack_base,
            events,
            dispatches,
            tail,
            break_at,
        } => cmd_trace(
            &input, va, steps, stack_base, events, dispatches, tail, break_at,
        ),
        Cmd::Cfg {
            input,
            va,
            stack_base,
            steps,
            max_blocks,
            events,
            state,
            timeout,
            vm_section,
        } => cmd_cfg(
            &input,
            va,
            stack_base,
            steps,
            max_blocks,
            events,
            state,
            timeout,
            &vm_section,
        ),
        Cmd::Vmctx {
            input,
            va,
            stack_base,
            steps,
            window,
        } => cmd_vmctx(&input, va, stack_base, steps, window),
        Cmd::Ssa {
            input,
            va,
            stack_base,
            steps,
            max_blocks,
            timeout,
            joins_only,
            vm_section,
        } => cmd_ssa(
            &input,
            va,
            stack_base,
            steps,
            max_blocks,
            timeout,
            joins_only,
            &vm_section,
        ),
        Cmd::WriteDevirt {
            input,
            output,
            steps,
            max_blocks,
            timeout,
            vm_section,
            jobs,
            cpu_percent,
            only_entry,
            patch_base,
        } => cmd_write_devirt(
            &input,
            &output,
            steps,
            max_blocks,
            timeout,
            &vm_section,
            jobs,
            cpu_percent,
            &only_entry,
            patch_base.as_deref(),
        ),
        Cmd::Devirt {
            input,
            va,
            output,
            stack_base,
            steps,
            max_blocks,
            timeout,
            dis,
        } => cmd_devirt(
            &input,
            va,
            output.as_deref(),
            stack_base,
            steps,
            max_blocks,
            timeout,
            dis,
        ),
        Cmd::Lower {
            input,
            va,
            stack_base,
            steps,
            max_blocks,
            timeout,
            vm_section,
        } => cmd_lower(
            &input,
            va,
            stack_base,
            steps,
            max_blocks,
            timeout,
            &vm_section,
        ),
        Cmd::Unresolved {
            input,
            va,
            stack_base,
            steps,
            max_blocks,
            timeout,
            json,
        } => cmd_unresolved(
            &input,
            va,
            stack_base,
            steps,
            max_blocks,
            timeout,
            json.as_deref(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::error::ErrorKind;

    #[test]
    fn write_devirt_accepts_cpu_percent_boundaries() {
        for percent in [1_u8, 60, 100] {
            let cli = Cli::try_parse_from([
                "tvm-devirt",
                "write-devirt",
                "input.dll",
                "output.dll",
                "--cpu-percent",
                &percent.to_string(),
            ])
            .unwrap();
            assert!(
                matches!(cli.cmd, Cmd::WriteDevirt { cpu_percent: Some(p), jobs: None, .. } if p == percent)
            );
        }
    }

    #[test]
    fn write_devirt_rejects_invalid_cpu_percent() {
        for percent in ["0", "101", "256", "-1", "1.5", "invalid"] {
            assert!(
                Cli::try_parse_from([
                    "tvm-devirt",
                    "write-devirt",
                    "input.dll",
                    "output.dll",
                    "--cpu-percent",
                    percent,
                ])
                .is_err(),
                "accepted {percent}"
            );
        }
    }

    #[test]
    fn write_devirt_cpu_percent_conflicts_with_jobs() {
        for options in [
            ["--cpu-percent", "60", "--jobs", "12"],
            ["--jobs", "12", "--cpu-percent", "60"],
        ] {
            let result = Cli::try_parse_from(
                ["tvm-devirt", "write-devirt", "input.dll", "output.dll"]
                    .into_iter()
                    .chain(options),
            );
            assert_eq!(result.err().unwrap().kind(), ErrorKind::ArgumentConflict);
        }
    }
}
