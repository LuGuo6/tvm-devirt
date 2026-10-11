r"""Reveal hash-bound PE entry/export functions in an already analyzed IDA database.

This is an IDA display preparation step, not a PE repair. It changes only the
FUNC_HIDDEN visibility bit on exact, executable PE entry/export function heads.
It does not create functions, change types/SP/other flags, suppress warnings,
patch bytes, or translate exception handlers.

In IDA's Python console:
    import runpy
    apply = runpy.run_path(r"PATH\scripts\prepare_ida.py")["apply"]
    apply(r"PATH\output.dll.layout.json", r"PATH\ida-preparation.json")

File > Script file also works when output.dll.layout.json is next to the input
DLL. A report is written beside the manifest. The input hash, loaded bytes,
image base, manifest ranges and PE entry/export table are checked before edits.
"""
from pathlib import Path
import hashlib
import json
import struct


def sha256(data):
    return hashlib.sha256(data).hexdigest().upper()


def pe_entries(data):
    """Read executable entry/export RVAs from an AMD64 PE32+ file, not IDA names."""
    def unpack(fmt, at):
        size = struct.calcsize(fmt)
        if at < 0 or at + size > len(data):
            raise ValueError("truncated PE structure")
        return struct.unpack_from(fmt, data, at)

    if data[:2] != b"MZ":
        raise ValueError("not a PE image")
    pe, = unpack("<I", 0x3c)
    if data[pe:pe + 4] != b"PE\0\0":
        raise ValueError("missing PE signature")
    machine, count = unpack("<HH", pe + 4)
    optional_size, = unpack("<H", pe + 20)
    opt = pe + 24
    magic, = unpack("<H", opt)
    if machine != 0x8664 or magic != 0x20b or optional_size < 112:
        raise ValueError("requires AMD64 PE32+")
    base, = unpack("<Q", opt + 24)
    entry, = unpack("<I", opt + 16)
    headers, = unpack("<I", opt + 60)
    directories, = unpack("<I", opt + 108)
    if directories > (optional_size - 112) // 8:
        raise ValueError("directory count exceeds optional header")
    sections = []
    for index in range(count):
        at = opt + optional_size + index * 40
        virtual, rva, raw_size, raw = unpack("<IIII", at + 8)
        flags, = unpack("<I", at + 36)
        if raw + raw_size > len(data):
            raise ValueError("truncated section")
        sections.append((rva, max(virtual, raw_size), raw_size, raw, flags))

    def offset(rva, size):
        if rva < 0 or size < 0 or rva + size > 0x100000000:
            raise ValueError("invalid RVA extent")
        choices = []
        if rva + size <= headers and rva + size <= len(data):
            choices.append(rva)
        choices.extend(raw + rva - lo for lo, _, raw_size, raw, _ in sections
                       if lo <= rva and rva + size <= lo + raw_size)
        if len(choices) != 1:
            raise ValueError("unbacked or ambiguous RVA")
        return choices[0]

    def executable(rva):
        return sum(lo <= rva < lo + size and bool(flags & 0x20000000)
                   for lo, size, _, _, flags in sections) == 1

    def cstring(rva):
        chars = bytearray()
        for index in range(65536):
            value = data[offset(rva + index, 1)]
            if not value:
                return chars.decode("ascii", errors="replace")
            chars.append(value)
        raise ValueError("unterminated export string")

    targets = {}
    if entry:
        if not executable(entry):
            raise ValueError("PE entry point is not executable")
        targets[entry] = ["PE entry point"]
    skipped_exports = []
    if directories:
        export_rva, export_size = unpack("<II", opt + 112)
        if bool(export_rva) != bool(export_size):
            raise ValueError("incomplete export directory")
        if export_rva:
            exp = offset(export_rva, max(export_size, 40))
            ordinal_base, functions, names, address_table, name_table, ordinal_table = unpack("<IIIIII", exp + 16)
            address_offset = offset(address_table, functions * 4)
            name_offset = offset(name_table, names * 4)
            ordinal_offset = offset(ordinal_table, names * 2)
            labels = {}
            for index in range(names):
                ordinal, = unpack("<H", ordinal_offset + index * 2)
                if ordinal >= functions:
                    raise ValueError("export ordinal outside address table")
                name_rva, = unpack("<I", name_offset + index * 4)
                labels.setdefault(ordinal, []).append(cstring(name_rva))
            for index in range(functions):
                rva, = unpack("<I", address_offset + index * 4)
                if not rva:
                    continue
                names_for_entry = labels.get(index, ["ordinal " + str(ordinal_base + index)])
                if export_rva <= rva < export_rva + export_size or not executable(rva):
                    skipped_exports.append({"rva": rva, "names": names_for_entry,
                                            "reason": "forwarded or non-executable export"})
                    continue
                targets.setdefault(rva, []).extend(names_for_entry)
    return base, entry, targets, skipped_exports, offset


def apply(manifest_path=None, report_path=None, f5_count=8):
    """Apply only verified display edits and return a before/after audit report."""
    import ida_bytes
    import ida_funcs
    import ida_frame
    import ida_hexrays
    import ida_ida
    import ida_idaapi
    import ida_lines
    import ida_nalt
    import idautils

    if not 0 <= f5_count <= 32:
        raise ValueError("f5_count must be between 0 and 32")
    manifest_path = Path(manifest_path or (ida_nalt.get_input_file_path() + ".layout.json")).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "tvm-devirt-layout" or manifest.get("version") != 2:
        raise ValueError("unsupported layout manifest")
    source = Path(ida_nalt.get_input_file_path()).resolve(strict=True)
    data = source.read_bytes()
    if manifest.get("output_size") != len(data) or manifest.get("output_sha256", "").upper() != sha256(data):
        raise ValueError("manifest does not match IDA input file")
    base, entry_rva, targets, skipped, offset = pe_entries(data)
    if base != manifest["image_base"] or base != ida_nalt.get_imagebase():
        raise ValueError("manifest, PE and IDB image bases differ; rebased databases are not edited")
    for item in manifest["ranges"]:
        rva, size = item["start_rva"], item["size"]
        raw = offset(rva, size)
        expected = data[raw:raw + size]
        if sha256(expected) != item["sha256"].upper() or ida_bytes.get_bytes(base + rva, size) != expected:
            raise ValueError("manifest range differs from input or loaded IDB bytes")

    # Preflight every function before any visibility change. Interior exports
    # and missing functions are recorded, never assigned guessed boundaries.
    functions = []
    unprepared = []
    for rva, names in sorted(targets.items()):
        va = base + rva
        function = ida_funcs.get_func(va)
        if function is None or function.start_ea != va:
            unprepared.append({"va": hex(va), "names": names,
                               "reason": "no exact function head; finish IDA analysis first"})
            continue
        size = function.end_ea - va
        raw = offset(rva, size)
        expected = data[raw:raw + size]
        if ida_bytes.get_bytes(va, size) != expected:
            raise ValueError("entry/export function bytes differ from input")
        functions.append((va, names, expected))

    def snapshot(va, expected):
        function = ida_funcs.get_func(va)
        stack = []
        for address in idautils.FuncItems(va):
            if ida_bytes.is_code(ida_bytes.get_flags(address)):
                delta = ida_frame.get_spd(function, address)
                stack.append((address, delta))
        return dict(start=hex(function.start_ea), end=hex(function.end_ea),
                    name=ida_funcs.get_func_name(va), flags=function.flags,
                    hidden=bool(function.flags & ida_funcs.FUNC_HIDDEN),
                    library=bool(function.flags & ida_funcs.FUNC_LIB),
                    chunks=list(idautils.Chunks(va)),
                    sp_sha256=sha256(json.dumps(stack).encode()),
                    positive_sp_count=sum(delta != ida_idaapi.BADADDR and delta > 0 for _, delta in stack),
                    unknown_sp_count=sum(delta == ida_idaapi.BADADDR for _, delta in stack),
                    bytes_sha256=sha256(ida_bytes.get_bytes(va, len(expected))))

    report = dict(schema="tvm-devirt-ida-preparation", version=1,
                  input=str(source), input_sha256=sha256(data),
                  manifest=str(manifest_path), manifest_sha256=sha256(manifest_path.read_bytes()),
                  auto_hide_library_functions=ida_ida.inf_hide_libfuncs(),
                  changes=[], f5=[], unprepared=unprepared, skipped_exports=skipped,
                  limits=["IDA visibility only; PE bytes, other function flags and SP are not modified.",
                          "F5 samples and static display checks do not establish runtime or exception equivalence."])
    report_path = Path(report_path) if report_path else manifest_path.with_suffix(".ida-preparation.json")
    try:
        for va, names, expected in functions:
            before = snapshot(va, expected)
            if before["hidden"]:
                ida_funcs.set_visible_func(ida_funcs.get_func(va), True)
            after = snapshot(va, expected)
            check = (after["flags"] == before["flags"] & ~ida_funcs.FUNC_HIDDEN
                     and after["bytes_sha256"] == before["bytes_sha256"]
                     and after["sp_sha256"] == before["sp_sha256"]
                     and after["chunks"] == before["chunks"]
                     and after["start"] == before["start"] and after["end"] == before["end"]
                     and after["name"] == before["name"] and not after["hidden"])
            report["changes"].append(dict(va=hex(va), names=names, before=before, after=after, passes=check))
            if not check:
                raise RuntimeError("visibility update changed unexpected function state")
        available = ida_hexrays.init_hexrays_plugin() if f5_count else False
        report["hexrays_available"] = available
        sample = sorted(functions, key=lambda row: (row[0] != base + entry_rva, row[0]))
        for va, _, expected in sample:
            if len(report["f5"]) >= f5_count:
                break
            if len(expected) > 65536:
                continue
            item = dict(va=hex(va), size=len(expected), success=False)
            try:
                cfunc = ida_hexrays.decompile(va) if available else None
                if cfunc:
                    text = "\n".join(ida_lines.tag_remove(line.line) for line in cfunc.get_pseudocode())
                    item.update(success=True, positive_sp_warning="positive sp value" in text.lower(),
                                text_sha256=sha256(text.encode("utf-8")))
                else:
                    item["error"] = "Hex-Rays unavailable or no decompilation result"
            except Exception as error:
                item["error"] = str(error)
            item["after"] = snapshot(va, expected)
            report["f5"].append(item)
        report["final_state"] = [dict(va=hex(va), **snapshot(va, expected)) for va, _, expected in functions]
        report["input_bytes_unchanged"] = source.read_bytes() == data
        report["all_visible"] = (not unprepared and bool(functions)
                                 and all(not item["hidden"] for item in report["final_state"]))
        report["sampled_f5_pass"] = (bool(report["f5"]) and all(item["success"] and not item.get("positive_sp_warning", True)
                                                                              for item in report["f5"])) if f5_count else None
        report["display_preparation_passes"] = (report["all_visible"] and report["input_bytes_unchanged"]
                                                 and all(item["passes"] for item in report["changes"]))
    finally:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    import ida_nalt
    import idc
    manifest = idc.ARGV[1] if len(idc.ARGV) > 1 else ida_nalt.get_input_file_path() + ".layout.json"
    result = apply(manifest)
    print(json.dumps({"manifest": manifest, "display_preparation_passes": result["display_preparation_passes"]}))


if __name__ == "__main__":
    main()
