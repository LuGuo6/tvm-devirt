# v9 集成版

本仓库把 `tvm-devirt-mirror` 中截至 v9 的累计 Rust 实现整合到同一个
`tvm-devirt` 可执行文件。原仓库历史保留，诊断、恢复、原生代码生成和 PE
写回直接使用同一套 IR。`v9` 是 ACE-Base64 样本的第九轮派生产物名称，
不是 TVM 的版本号，也不是“任意 DLL 全部恢复”的保证。

## 构建

需要 Rust stable（支持 edition 2024）。Windows 使用 MSVC 工具链，并在
Visual Studio 的 **x64 Native Tools Command Prompt** 中运行：

```text
cargo test --release --locked
cargo build --release --locked
```

生成文件：`target\release\tvm-devirt.exe`。如果 `cargo` 未加入 PATH，
将 Rust 安装目录下的 `.cargo\bin` 加入当前终端的 PATH。

## 使用

以下命令在 PowerShell 中从仓库根目录运行。输入支持 AMD64 PE32+ 的
EXE、DLL 和 SYS；工具读取文件进行静态分析，不启动目标程序。

```powershell
# 查看布局和发现的入口。
.\target\release\tvm-devirt.exe info .\input.dll
.\target\release\tvm-devirt.exe entries .\input.dll --all

# 对完整输入进行恢复；需要时调整资源预算。
.\target\release\tvm-devirt.exe write-devirt .\input.dll .\output.dll --max-blocks 1024 --jobs 2

# 单个入口的 CFG、未解决原因和原生代码。
.\target\release\tvm-devirt.exe cfg .\input.dll 0x180001000
.\target\release\tvm-devirt.exe unresolved .\input.dll 0x180001000 --json .\unresolved.json
.\target\release\tvm-devirt.exe devirt .\input.dll 0x180001000 --output .\function.bin --dis

# 在已有派生文件上增量追加；第一个参数始终为原始输入。
.\target\release\tvm-devirt.exe write-devirt .\input.dll .\next.dll --patch-base .\output.dll --only-entry 0x180001000,0x180002000 --max-blocks 1024
```

请用自己的入口地址替换示例地址。增量输出使用新文件名。`--only-entry`
可重复或使用逗号分隔，地址必须来自原始输入的入口发现结果。`--patch-base`
检查映像布局与 VM 节兼容性；它不是完整文件身份或语义一致性的证明，
基线应由同一份原始文件生成并保留其来源和校验记录。

生成代码追加到 PE 最后一个节并设置执行属性，节名可能仍显示为 `.reloc`。
增量更新不会重定向旧生成函数中已经指向旧 body 的直接调用；重生成已有
入口时还需审查这些引用，或者从原始输入重新构建整份派生文件。

`--steps`、`--max-blocks`、`--timeout` 控制恢复预算，`--jobs` 控制批处理
并发数。提高预算不保证处理未知 VM 语义。`unresolved` 的诊断耗时不计入
恢复时间预算。

## 合入的实现

- 镜像的 `unresolved` 命令、每块失败原因、表达式图和 arena compaction 诊断。
- 内部 VM CALL/RET 继续执行、受限入口分支探索、guest context/VIP 识别。
- 算术与已知位化简，保守的 switch 值域传播、SSA 根保留和 join 处理。
- 符号栈按字节转发；未知别名、调用、boxed 内存副作用及 join 会清除缓存。
- 完整字节匹配的 stack-probe helper，保留有序页面读取副作用。
- 严格的 native vtable、单 helper wrapper 和完整 VM exit-tail 证明，包括
  v9 新增的 `+0x20` slot。仅观察到符号跳转或没有 VIP 不足以认定原生尾调用。
- `IndirectTailCall` IR、单次目标读取、helper 返回值传递、寄存器恢复、
  REX.W 尾跳与 unwind 生成。
- 不完整 CFG 在调度/寄存器分配之前拒绝，支持指定入口及增量 PE 写回。

集成时去除了研究工作树中的四个样本绝对地址 noreturn 特判。普通函数即使
位于相同地址也应正常返回；现在使用实际导入/指令证据。相邻入口的五字节
补丁还会在生成目标代码和调用重定向之前检查，拒绝会覆盖其他入口或跨越
已知函数/节边界的写回。因此覆盖数量可能与历史批次不同。

## 历史 v9 与验证边界

移植来源为镜像 `c18c4524efbf573fd23199fb7ad38b15fd556d72` 加上当时的
v9 累计源码修改；目标仓库基线为
`1639a003e2b3c5220b4c8cb6d67cc0c06903175a`。保留上游及镜像作者归属。
样本、历史派生 DLL 和 IDA 数据库不作为源码依赖。

历史记录如下：

| 项目 | 已记录结果 |
| --- | --- |
| 原始 ACE-Base64.dll SHA256 | `82A07DB52E097929F18BF14E4D276C6A8F790EDA4EF2B1FABC9CCE690F12C64F` |
| 历史 v9 DLL SHA256 | `57807FDCB3EEE68D1473BD27AE4398DFA9E8C143F3B1ED0D698DB6E4BAFF8752` |
| 累计入口重定向 | 1290 / 1453，含扫描别名，非独立源函数数量 |
| v9 新增入口 | `0x180535A80`，37 步、1 块、26 字节代码 |
| 当时的 Rust 测试 | 260 通过 |
| 当时新增函数的 Windows unwind 检查 | 4 / 4 通过 |

历史 v9 是 v2–v8 多次增量结果的累计产物，部分早期轮次还使用样本专用
后处理和独立审核。合入源码不等于从原始 DLL 单次运行便逐字节重建历史
v9。老函数也不会因为追加新函数就自动重新生成。历史 v9 仍有 163 个未恢复
候选入口、15 条已生成代码到 VM 的直接跳转，以及间接/传递 VM 依赖。

PE 结构、单元测试、局部等价和 unwind 检查各自覆盖不同内容；它们没有证明
整个 DLL 的运行兼容性或全部函数的语义等价。产物用于分析。

## IDA 的 positive SP 警告

`positive sp value has been detected, the output may be wrong!` 表示反编译器
在函数路径上计算出的栈变化不一致或超过预期。可能涉及函数边界、错误的
callee 类型/栈清理量、间接尾跳或复杂 RSP 恢复，也可能是生成代码错误。
它不是“仍有 TVM”的独立判据，也不应靠全局清除警告来视作修复。

核查时记录函数地址，先确定其位于原始 `.text`、VM 节还是新增 native
代码，再对照函数入口、每次 RSP 修改、CALL、RET/尾跳与对应 unwind 记录。
出现警告的伪代码需结合汇编验证；历史 v9 的局部验证不替这些函数背书。

本次核查已在历史 v9 继承的五个早期生成 body 中发现实际 RSP 漂移。
例如 `0x1845DF640`（原入口 `0x1811E6D40`）在 `sub rsp,0x188` 后出现两次
额外的 `lea rsp,[rsp+0xd0]`，末尾又 `add rsp,0x188`，相关返回路径净偏移
为 `+0x1a0`。单次新增 v9 函数的验证没有覆盖这些老 body，警告应逐函数分类。
