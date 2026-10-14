"""Narrow non-executing TorchVersion-to-str rewrite for owned PCOD ZIPs only."""
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import pickletools
import re
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAX_ARCHIVE = 1024**3
MAX_METADATA = 1024**2
MAX_OPS = 100_000
GLOBALS = {'collections OrderedDict', 'torch FloatStorage', 'torch LongStorage',
           'torch._utils _rebuild_tensor_v2', 'torch.torch_version TorchVersion'}


def need(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream: return stream_sha(stream)


def stream_sha(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024**2), b''): result.update(block)
    return result.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


store = load('pcod54_tensor_primitives52', ROOT / '52_pgsnet_streamed_checkpoint/tensor_store.py')
read, write = store.read_json, store.write_json


def operations(payload):
    need(type(payload) is bytes and 0 < len(payload) <= MAX_METADATA, 'Bounded pickle instruction bytes')
    result = []
    for item in pickletools.genops(payload):
        result.append(item)
        need(len(result) <= MAX_OPS, 'Instruction count cap')
    need(result[0][0].name == 'PROTO' and result[0][1] == 2, 'Only observed protocol2 layout')
    need(result[-1][0].name == 'STOP' and result[-1][2] + 1 == len(payload), 'Exact STOP/no trailing bytes')
    return result


def normalize_metadata(payload, expected_version):
    need(type(expected_version) is str and re.fullmatch(r'\d+\.\d+\.\d+\+cpu', expected_version), 'Exact CPU version string')
    ops = operations(payload)
    need(not any(x[0].name in ('STACK_GLOBAL', 'EXT1', 'EXT2', 'EXT4', 'INST', 'OBJ') for x in ops), 'Unsupported object opcode')
    global_names = {arg for op, arg, _ in ops if op.name == 'GLOBAL'}
    need(global_names <= GLOBALS, 'Unexpected class/function global in owned artifact')
    positions = [i for i, (op, arg, _) in enumerate(ops) if op.name == 'GLOBAL' and arg == 'torch.torch_version TorchVersion']
    need(len(positions) == 1, 'Exactly one observed TorchVersion constructor')
    i = positions[0]; part = ops[i:i+9]
    need(len(part) == 9, 'Truncated version constructor')
    names = [op.name for op, _, _ in part]
    need(names[0] == 'GLOBAL' and names[2] == 'BINUNICODE' and names[4] == 'TUPLE1'
         and names[6] == 'NEWOBJ' and names[1] in ('BINPUT', 'LONG_BINPUT')
         and names[3] in ('BINPUT', 'LONG_BINPUT') and names[5] in ('BINPUT', 'LONG_BINPUT')
         and names[7] in ('BINPUT', 'LONG_BINPUT'), 'Unexpected version constructor layout')
    need(part[2][1] == expected_version, 'Version string differs from frozen identity')
    need(sum(op.name == 'NEWOBJ' for op, _, _ in ops) == 1, 'Additional object constructor')
    removed_memo = {part[1][1], part[5][1]}
    need(len(removed_memo) == 2, 'Aliased constructor memo')
    need(not any(op.name in ('GET', 'BINGET', 'LONG_BINGET') and arg in removed_memo for op, arg, _ in ops),
         'Removed class/tuple memo is referenced elsewhere')
    # Preserve literal-string and result memo IDs. No other instruction is changed.
    spans = [(part[0][2], part[2][2]), (part[4][2], part[7][2])]
    normalized = payload[:spans[0][0]] + payload[spans[0][1]:spans[1][0]] + payload[spans[1][1]:]
    remaining = operations(normalized)
    need(not any(op.name == 'GLOBAL' and arg == 'torch.torch_version TorchVersion' for op, arg, _ in remaining), 'Class removal failed')
    return normalized, {'original_metadata_sha256': hashlib.sha256(payload).hexdigest(),
        'normalized_metadata_sha256': hashlib.sha256(normalized).hexdigest(),
        'removed_byte_ranges': spans, 'removed_bytes_hex': [payload[a:b].hex() for a, b in spans],
        'removed_unreferenced_memos': sorted(removed_memo), 'literal_version': expected_version,
        'string_memo': part[3][1], 'result_memo': part[7][1], 'other_instruction_bytes_unchanged': True,
        'before_globals': sorted(global_names), 'normalization': 'string_subclass_to_builtin_string_only'}


def archive_layout(archive):
    infos = archive.infolist()
    need(4 <= len(infos) <= 4096 and len({x.filename for x in infos}) == len(infos), 'Bounded unique ZIP members')
    need(sum(x.file_size for x in infos) <= MAX_ARCHIVE, 'Uncompressed archive cap')
    prefixes = {PurePosixPath(x.filename).parts[0] for x in infos if PurePosixPath(x.filename).parts}
    need(len(prefixes) == 1, 'Single archive root required')
    prefix = next(iter(prefixes))
    need(re.fullmatch(r'[A-Za-z0-9_.-]+', prefix) is not None and prefix not in ('.', '..'), 'Canonical archive root')
    suffixes = set()
    for entry in infos:
        path = PurePosixPath(entry.filename)
        need(not path.is_absolute() and '\\' not in entry.filename and '..' not in path.parts
             and not entry.is_dir() and not entry.flag_bits & 1 and entry.compress_type == zipfile.ZIP_STORED,
             'Only bounded unencrypted stored regular ZIP entries')
        need((entry.external_attr >> 16) & 0o170000 != 0o120000, 'Symlink member rejected')
        suffix = '/'.join(path.parts[1:]); suffixes.add(suffix)
        need(suffix in ('data.pkl', 'byteorder', 'version', '.data/serialization_id')
             or re.fullmatch(r'data/[0-9]+', suffix) is not None, 'Unexpected checkpoint ZIP member')
    need({'data.pkl', 'byteorder', 'version', '.data/serialization_id'} <= suffixes, 'Incomplete checkpoint archive')
    need(archive.getinfo(prefix + '/data.pkl').file_size <= MAX_METADATA, 'Pickle metadata cap')
    need(archive.read(prefix + '/byteorder') == b'little', 'Only native little endian PCOD checkpoints')
    need(archive.read(prefix + '/version') == b'3\n', 'Only observed torch archive version3')
    return infos, prefix + '/data.pkl'


def normalize_archive(source, output, expected_sha, version):
    source, output = Path(source), Path(output)
    need(source.resolve() != output.resolve(), 'Never overwrite source checkpoint')
    need(source.stat().st_size <= MAX_ARCHIVE and sha(source) == expected_sha, 'Owned source size/hash differs')
    original_members = {}
    with zipfile.ZipFile(source) as src:
        infos, metadata_name = archive_layout(src)
        new_metadata, details = normalize_metadata(src.read(metadata_name), version)
        # A new archive only; failure leaves its bytes for inspection.
        with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as dst:
            for entry in infos:
                with src.open(entry) as stream: original_members[entry.filename] = stream_sha(stream)
                if entry.filename == metadata_name:
                    dst.writestr(entry.filename, new_metadata)
                else:
                    with src.open(entry) as a, dst.open(entry.filename, 'w', force_zip64=True) as b:
                        for block in iter(lambda: a.read(1024**2), b''): b.write(block)
    with zipfile.ZipFile(output) as dst:
        copied, copied_metadata = archive_layout(dst)
        need([x.filename for x in copied] == [x.filename for x in infos] and copied_metadata == metadata_name, 'Member order/name drift')
        derived_members = {}
        for entry in copied:
            with dst.open(entry) as stream: derived_members[entry.filename] = stream_sha(stream)
        need(all(derived_members[name] == digest for name, digest in original_members.items() if name != metadata_name),
             'Non-metadata member bytes changed')
        need(dst.read(metadata_name) == new_metadata, 'Normalized metadata changed during serialization')
    need(sha(source) == expected_sha, 'Source changed during conversion')
    return {'source_sha256': expected_sha, 'normalized_sha256': sha(output), 'metadata_member': metadata_name,
            'original_members_sha256': original_members, 'normalized_members_sha256': derived_members,
            'metadata_rewrite': details, 'storage_and_tensor_recipe_bytes_unchanged': True,
            'source_file_unchanged': True, 'pickle_executed_by_normalizer': False}
