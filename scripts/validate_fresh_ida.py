"""Validate a manifest-bound DLL in a fresh, isolated idalib database.

Run with D:\Program Files\IDA Pro\python311\python.exe. No existing IDB is
opened, and no target code is executed. The new copy and .i64 are retained.
No byte/SP/function-flag patches or global warning suppression are applied.
"""
from pathlib import Path
import argparse
import bisect
import hashlib
import json
import shutil
import struct
import time


def digest(data):
    return hashlib.sha256(data).hexdigest().upper()


def pe_layout(data):
    assert data[:2] == b'MZ'
    pe = struct.unpack_from('<I', data, 0x3c)[0]
    assert data[pe:pe + 4] == b'PE\0\0'
    machine, count = struct.unpack_from('<HH', data, pe + 4)
    optional_size = struct.unpack_from('<H', data, pe + 20)[0]
    opt = pe + 24
    assert machine == 0x8664 and struct.unpack_from('<H', data, opt)[0] == 0x20b
    base = struct.unpack_from('<Q', data, opt + 24)[0]
    entry = struct.unpack_from('<I', data, opt + 16)[0]
    size = struct.unpack_from('<I', data, opt + 56)[0]
    sections = []
    for i in range(count):
        at = opt + optional_size + 40 * i
        virtual_size, rva, raw_size, raw = struct.unpack_from('<IIII', data, at + 8)
        sections.append(dict(name=data[at:at + 8].rstrip(b'\0').decode(errors='replace'),
            rva=rva, virtual_size=virtual_size, raw=raw, raw_size=raw_size,
            flags=struct.unpack_from('<I', data, at + 36)[0]))
    return base, entry, size, sections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--analysis-seconds', type=float, default=300)
    parser.add_argument('--f5-count', type=int, default=8)
    args = parser.parse_args()
    source = args.input.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    data = source.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['schema'] == 'tvm-devirt-layout' and manifest['version'] == 2
    assert manifest['output_sha256'].upper() == digest(data)
    assert manifest['output_size'] == len(data)
    base, ep, image_size, sections = pe_layout(data)
    assert base == manifest['image_base']
    assert args.analysis_seconds > 0 and 0 <= args.f5_count <= 32
    ranges = manifest['ranges']
    for item in ranges:
        lo, length = item['start_rva'], item['size']
        matches = [s for s in sections if s['rva'] <= lo and lo + length <= s['rva'] + s['raw_size']]
        assert len(matches) == 1, (item, 'ambiguous/unbacked range')
        section = matches[0]
        raw = section['raw'] + lo - section['rva']
        assert digest(data[raw:raw + length]) == item['sha256'].upper()
    out = args.output_dir.resolve()
    # Refuse a previous result directory, even if it contains only a stale DB.
    out.mkdir(parents=True, exist_ok=False)
    binary = out / 'candidate.dll'
    shutil.copyfile(source, binary)
    (out / 'input-manifest.json').write_bytes(manifest_path.read_bytes())
    report_path = out / 'ida-validation.json'
    report = dict(input=str(source), input_sha256=digest(data), manifest_sha256=digest(manifest_path.read_bytes()),
        copy=str(binary), state='opening-fresh-idb', checks={}, bodies=[], f5=[], limits=[
            'IDA static analysis and sampled F5 do not prove DLL runtime or exception equivalence.',
            'Incomplete autoanalysis is explicitly reported; unknown SP or missing functions never count as passes.',
            'Per-step analysis budgets do not interrupt a single long IDA callback; run as a monitored process.'])

    def save():
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    save()
    # idapro must be imported before any IDAPython module.
    import idapro
    import ida_auto
    import ida_bytes
    import ida_funcs
    import ida_frame
    import ida_hexrays
    import ida_idaapi
    import ida_ida
    import ida_idp
    import ida_lines
    import ida_nalt
    import ida_segment
    import ida_ua
    import idautils
    report['idalib_version'] = idapro.get_library_version()
    opened = False
    try:
        status = idapro.open_database(str(binary), False)
        assert status == 0, ('open_database', status)
        opened = True
        assert Path(ida_nalt.get_input_file_path()).resolve() == binary
        report['state'] = 'analyzing-fresh-idb'
        save()
        code = [r for r in ranges if r['kind'] == 'code' and r['state'] == 'active']
        metadata = [r for r in ranges if r['kind'] in ('metadata-section', 'runtime-table', 'unwind')]
        # Loader/runtime-table analysis should discover these functions. Queue
        # missing beginnings without forcing their boundaries or stack deltas.
        for item in code:
            ida_auto.auto_make_proc(base + item['start_rva'])
        ida_auto.auto_make_proc(base + ep)
        deadline = time.monotonic() + args.analysis_seconds
        work = [(base + ep, base + ep + 0x100)] + [(base + r['start_rva'], base + r['start_rva'] + r['size']) for r in code]
        pending = list(work)
        steps = 0
        last_progress = time.monotonic()
        while pending and time.monotonic() < deadline:
            next_pending = []
            for lo, hi in pending:
                if time.monotonic() >= deadline:
                    next_pending.append((lo, hi))
                    continue
                if ida_auto.auto_make_step(lo, hi):
                    next_pending.append((lo, hi))
                    steps += 1
            pending = next_pending
            if time.monotonic() - last_progress >= 10:
                report['analysis_progress'] = dict(steps=steps, pending_ranges=len(pending))
                save()
                last_progress = time.monotonic()
        report['analysis'] = dict(steps=steps, requested_ranges=len(work), pending_ranges=pending,
            all_global_queues_empty=bool(ida_auto.auto_is_ok()), scoped_queues_drained=not pending)
        function = ida_funcs.get_func(base + ep)
        report['entrypoint'] = None if not function else dict(va=hex(base + ep), start=hex(function.start_ea),
            end=hex(function.end_ea), name=ida_funcs.get_func_name(function.start_ea), flags=hex(function.flags),
            hidden=bool(function.flags & ida_funcs.FUNC_HIDDEN), far=bool(function.flags & ida_funcs.FUNC_FAR))
        report['checks']['entry_function_exists'] = function is not None and function.start_ea == base + ep
        report['checks']['entry_visible'] = function is not None and not function.flags & ida_funcs.FUNC_HIDDEN
        report['checks']['source_and_copy_hash_equal'] = digest(binary.read_bytes()) == digest(data)
        report['checks']['manifest_ranges_match_ida_bytes'] = all(
            digest(ida_bytes.get_bytes(base + r['start_rva'], r['size']) or b'') == r['sha256'].upper() for r in ranges)
        permission_errors = []
        for item in ranges:
            if item['state'] != 'active' or item['kind'] not in ('code-section', 'metadata-section'):
                continue
            segment = ida_segment.getseg(base + item['start_rva'])
            expected = ida_segment.SEGPERM_READ | (ida_segment.SEGPERM_EXEC if item['kind'] == 'code-section' else 0)
            if segment is None or segment.perm != expected:
                permission_errors.append(dict(range=item, actual=None if segment is None else segment.perm, expected=expected))
        report['permission_errors'] = permission_errors
        report['checks']['split_code_metadata_permissions'] = not permission_errors
        merged_metadata = []
        for start, end in sorted((base + r['start_rva'], base + r['start_rva'] + r['size']) for r in metadata):
            if merged_metadata and start <= merged_metadata[-1][1]:
                merged_metadata[-1][1] = max(merged_metadata[-1][1], end)
            else:
                merged_metadata.append([start, end])
        metadata_ends = [r[1] for r in merged_metadata]
        metadata_functions = []
        for ea in idautils.Functions():
            for lo, hi in idautils.Chunks(ea):
                index = bisect.bisect_right(metadata_ends, lo)
                if index < len(merged_metadata) and merged_metadata[index][0] < hi:
                    metadata_functions.append(dict(function=hex(ea), chunk=[hex(lo), hex(hi)], metadata=merged_metadata[index]))
        report['metadata_functions'] = metadata_functions
        report['checks']['metadata_has_no_function_chunks'] = not metadata_functions
        for item in code:
            lo, hi = base + item['start_rva'], base + item['start_rva'] + item['size']
            function = ida_funcs.get_func(lo)
            body = dict(start=hex(lo), end=hex(hi), size=item['size'], function=None, positive_sp=[], unknown_sp=[], decode_errors=[], ownership_errors=[], instructions=0,
                unreachable_alignment=[], indirect_exits=[], reachable_decode_errors=[])
            if function:
                body['function'] = dict(start=hex(function.start_ea), end=hex(function.end_ea), flags=hex(function.flags),
                    sp_ready=bool(function.flags & ida_funcs.FUNC_SP_READY), chunks=[[hex(a),hex(b)] for a,b in idautils.Chunks(function.start_ea)])
            body['chunks_escape_body'] = [] if not function else [[hex(a),hex(b)] for a,b in idautils.Chunks(function.start_ea) if a < lo or b > hi]
            # Reachability is used only to distinguish disconnected CC/NOP
            # padding. Direct CALL targets remain separate callee functions.
            # An unresolved indirect jump disables the padding exemption.
            reachable = set()
            todo = [lo]
            while todo:
                address = todo.pop()
                if address in reachable or not lo <= address < hi:
                    continue
                reachable.add(address)
                instruction = ida_ua.insn_t()
                length = ida_ua.decode_insn(instruction, address)
                if not length or address + length > hi:
                    body['reachable_decode_errors'].append(hex(address))
                    continue
                call = ida_idp.is_call_insn(instruction)
                if not call:
                    if ida_idp.is_indirect_jump_insn(instruction):
                        body['indirect_exits'].append(hex(address))
                    for operand in instruction.ops:
                        if operand.type == ida_ua.o_near:
                            todo.append(operand.addr)
                if not instruction.get_canon_feature() & ida_idp.CF_STOP:
                    todo.append(address + length)
            body['reachable_instructions'] = len(reachable)
            cursor = lo
            while cursor < hi:
                instruction = ida_ua.insn_t()
                length = ida_ua.decode_insn(instruction, cursor)
                if not length or cursor + length > hi:
                    body['decode_errors'].append(hex(cursor)); break
                owner = ida_funcs.get_func(cursor)
                alignment = cursor not in reachable and not body['indirect_exits'] and length == 1 and ida_bytes.get_byte(cursor) in (0xcc, 0x90)
                if alignment and (owner is None or not ida_bytes.is_code(ida_bytes.get_full_flags(cursor))):
                    body['unreachable_alignment'].append(hex(cursor))
                    body['instructions'] += 1
                    cursor += length
                    continue
                if owner is None or owner.start_ea != lo or not ida_bytes.is_code(ida_bytes.get_full_flags(cursor)):
                    body['ownership_errors'].append(hex(cursor))
                if owner:
                    sp = ida_frame.get_spd(owner, cursor)
                    if sp in (ida_idaapi.BADADDR, -(1 << 63)):
                        body['unknown_sp'].append(hex(cursor))
                    elif sp > 0:
                        body['positive_sp'].append(dict(ea=hex(cursor), sp=sp))
                else:
                    body['unknown_sp'].append(hex(cursor))
                body['instructions'] += 1
                cursor += length
            body['passes'] = bool(function and function.start_ea == lo and function.flags & ida_funcs.FUNC_SP_READY
                and not body['positive_sp'] and not body['unknown_sp'] and not body['decode_errors'] and not body['ownership_errors'] and not body['reachable_decode_errors'] and not body['chunks_escape_body'])
            report['bodies'].append(body)
            if len(report['bodies']) % 25 == 0:
                report['body_progress'] = dict(done=len(report['bodies']), total=len(code))
                save()
        report['checks']['all_generated_bodies_sp_and_ownership'] = all(b['passes'] for b in report['bodies'])
        save()
        hexrays = bool(ida_hexrays.init_hexrays_plugin())
        report['hexrays_available'] = hexrays
        candidates = [r for r in code if r['size'] <= 65536]
        # Mix small/large candidates deterministically without touching huge bodies.
        candidates.sort(key=lambda r: (r['size'], r['start_rva']))
        selected = []
        if candidates and args.f5_count:
            for i in range(min(args.f5_count, len(candidates))):
                selected.append(candidates[i * (len(candidates) - 1) // max(1, min(args.f5_count, len(candidates)) - 1)])
        entry_function = ida_funcs.get_func(base + ep)
        if args.f5_count and entry_function and 0 < entry_function.end_ea - entry_function.start_ea <= 65536 and not any(r['start_rva'] == ep for r in selected):
            selected.insert(0, dict(start_rva=ep, size=entry_function.end_ea-entry_function.start_ea, kind='pe-entrypoint'))
        for item in selected:
            ea = base + item['start_rva']
            result = dict(va=hex(ea), size=item['size'], success=False)
            if hexrays:
                try:
                    cfunc = ida_hexrays.decompile(ea)
                    assert cfunc is not None
                    text = '\n'.join(ida_lines.tag_remove(line.line) for line in cfunc.get_pseudocode())
                    (out / ('f5-' + hex(ea) + '.c')).write_text(text, encoding='utf-8')
                    result['success'] = True
                    result['positive_sp_warning'] = 'positive sp value' in text.lower()
                    result['text_sha256'] = digest(text.encode())
                except Exception as error:
                    result['error'] = repr(error)
            else:
                result['error'] = 'Hex-Rays initialization failed'
            report['f5'].append(result)
            save()
        report['checks']['sampled_f5_pass'] = all(r['success'] and not r.get('positive_sp_warning') for r in report['f5']) and (not selected or hexrays)
        entry_function = ida_funcs.get_func(base + ep)
        report['entrypoint_after_f5'] = None if not entry_function else dict(name=ida_funcs.get_func_name(base + ep),
            flags=hex(entry_function.flags), hidden=bool(entry_function.flags & ida_funcs.FUNC_HIDDEN))
        report['checks']['entry_visible_after_f5'] = entry_function is not None and not entry_function.flags & ida_funcs.FUNC_HIDDEN
        report['checks']['no_pe_bytes_changed'] = digest(source.read_bytes()) == digest(data) == digest(binary.read_bytes())
        report['state'] = 'completed'
        report['all_checks_pass'] = all(report['checks'].values()) and not pending
        save()
    except Exception as error:
        report['state'] = 'failed'
        report['error'] = repr(error)
        save()
        raise
    finally:
        if opened:
            idapro.close_database(True)
        report['database_files'] = [str(p) for p in out.glob('candidate*.i*')]
        save()
    print(json.dumps(dict(report=str(report_path), state=report['state'], all_checks_pass=report.get('all_checks_pass'))))


if __name__ == '__main__':
    main()
