"""CPU-only conversion of prespecified local completed PCOD best checkpoints."""
import ctypes
from datetime import datetime, timezone
import gc
from pathlib import Path
import sys
import metadata54 as m
import tensor54 as s


def runtime():
    sys.path.insert(0, str(m.ROOT / '31_pgsnet_checkpoint_safety'))
    import guarded_inspection
    torch = guarded_inspection.require_runtime()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    m.need(kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000) != 0, 'Below-normal CPU priority required')
    return torch


def memory_guard():
    class Memory(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in
            ('total', 'available', 'page_total', 'page_available', 'virtual_total', 'virtual_available', 'extended')]
    value = Memory(); value.length = ctypes.sizeof(value)
    m.need(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)) != 0, 'RAM query failed')
    m.need(value.available >= 8 * 1024**3, 'Keep8GiB available physical RAM')
    return value.available


def plain(value):
    if type(value) is dict:
        m.need(all(type(k) is str for k in value), 'Plain JSON string keys')
        for x in value.values(): plain(x)
    elif type(value) in (list, tuple):
        for x in value: plain(x)
    else:
        m.need(type(value) in (str, int, float, bool, type(None)), 'Only plain metadata values')
    m.store.json_bytes(value)


def validate_best(torch, best, item):
    m.need(type(best) is dict and set(best) == {'identity', 'epoch', 'model_state', 'validation_iou'}, 'Exact best checkpoint fields')
    plain(best['identity'])
    m.need(best['identity'] == item['identity'] and type(best['epoch']) is int and best['epoch'] == item['best_epoch']
           and type(best['validation_iou']) is float and best['validation_iou'] == item['best_validation_iou'],
           'Frozen identity/validation-selected metadata differs')
    state = best['model_state']
    m.need(isinstance(state, dict) and 0 < len(state) <= 10000, 'Bounded state mapping')
    for key, value in state.items():
        m.need(type(key) is str and type(value) is torch.Tensor and value.device.type == 'cpu'
               and value.layout == torch.strided and value.is_contiguous()
               and value.dtype in (torch.float32, torch.int64) and torch.isfinite(value).all().item(),
               'Dense finite plain FP32/int64 CPU tensor only')
    extra = getattr(state, '_metadata', None)
    if extra is not None:
        # OrderedDict is encoded as the same ordered key/value JSON mapping, not pickled.
        extra = dict(extra); plain(extra)
    return {'identity': best['identity'], 'epoch': best['epoch'],
            'validation_iou': best['validation_iou'], 'state_dict_metadata': extra}


def validate_plan(plan):
    m.need(plan['external_checkpoint_sources_allowed'] is False and len(plan['checkpoints']) == 3,
           'Only this fixed completed local prefix is admitted')
    for path, digest in plan['source_sha256'].items(): m.need(m.sha(path) == digest, 'Bound source changed: ' + path)
    m.need(m.sha(m.ROOT / '25_pcod_portable_runner/formal_protocol.json') == plan['formal_protocol_sha256'], 'Formal protocol changed')


def main():
    plan = m.read(m.HERE / 'plan.json'); validate_plan(plan)
    tests = m.read(m.HERE / 'cpu_attempt_v1/cpu_test_results.json')
    m.need(tests['passed'] and tests['plan_sha256'] == m.sha(m.HERE / 'plan.json'), 'Bound tests must pass first')
    torch = runtime()
    before = list(torch.serialization.get_safe_globals())
    output = m.HERE / 'conversions'; output.mkdir(exist_ok=False)
    completed = []
    for item in plan['checkpoints']:
        available = memory_guard(); validate_plan(plan)
        source = Path(item['output_dir'])
        for name, digest in item['artifacts_sha256'].items(): m.need(m.sha(source / name) == digest, 'Original artifact changed')
        m.need(m.sha(source / 'portable_complete.json') == item['completion_sha256'], 'Original completion changed')
        target = output / item['run_id']; target.mkdir(exist_ok=False)
        try:
            normalized = target / 'normalized_best.pt'
            normalization = m.normalize_archive(source / 'best.pt', normalized,
                item['artifacts_sha256']['best.pt'], item['identity']['versions']['torch'])
            m.write(target / 'normalization.json', normalization)
            m.need(list(torch.serialization.get_safe_globals()) == before, 'Allowlist drift before load')
            best = torch.load(normalized, map_location='cpu', weights_only=True)
            m.need(list(torch.serialization.get_safe_globals()) == before, 'Allowlist drift during load')
            metadata = validate_best(torch, best, item)
            binding = {'run_id': item['run_id'], 'original_best_sha256': item['artifacts_sha256']['best.pt'],
                       'original_completion_sha256': item['completion_sha256'], 'formal_protocol_sha256': plan['formal_protocol_sha256']}
            receipt = s.write_state(target / 'state', best['model_state'], metadata, binding)
            rebuilt = s.load_state(target / 'state', receipt, binding)
            m.need(list(rebuilt['model_state']) == list(best['model_state']), 'State key order changed')
            for key, value in best['model_state'].items():
                other = rebuilt['model_state'][key]
                m.need(value.dtype == other.dtype and value.shape == other.shape and torch.equal(value, other), 'Exact tensor roundtrip changed')
            m.need(rebuilt['identity'] == best['identity'] and rebuilt['epoch'] == best['epoch']
                   and rebuilt['validation_iou'] == best['validation_iou'], 'Metadata roundtrip changed')
            manifest = s.verify_state(target / 'state', receipt, binding)
            validate_plan(plan)
            m.need(m.sha(source / 'best.pt') == item['artifacts_sha256']['best.pt'], 'Original best changed')
            report = {'passed': True, 'utc': datetime.now(timezone.utc).isoformat(), 'binding': binding,
                'normalized_sha256': m.sha(normalized), 'normalization_sha256': m.sha(target / 'normalization.json'),
                'state_receipt_sha256': receipt, 'tensors': len(best['model_state']),
                'tensor_bytes': manifest['total_tensor_bytes'], 'maximum_tensor_bytes': max(x['bytes'] for x in manifest['schema']),
                'all_tensor_shape_dtype_value_roundtrip_equal': True, 'non_metadata_archive_members_unchanged': True,
                'torch_runtime': str(torch.__version__), 'initial_available_memory_bytes': available,
                'weights_only': True, 'user_allowlist_added': False, 'unsafe_fallback': False,
                'full_model_created': False, 'device_or_test_inference': False, 'source_original_unmodified': True,
                'not_a_generic_untrusted_pickle_sanitizer': True}
            m.write(target / 'conversion_report.json', report)
            m.write(target / 'conversion_complete.json', {'report_sha256': m.sha(target / 'conversion_report.json'),
                                                        'plan_sha256': m.sha(m.HERE / 'plan.json')})
            completed.append({'run_id': item['run_id'], 'report_sha256': m.sha(target / 'conversion_report.json')})
            print({'completed': item['run_id'], 'tensors': report['tensors'], 'tensor_bytes': report['tensor_bytes']}, flush=True)
            del best, rebuilt, manifest
            gc.collect()
        except BaseException as error:
            m.write(target / 'failure.json', {'error': type(error).__name__ + ': ' + str(error), 'automatic_retry': False})
            raise
    m.write(output / 'all_completed.json', {'completed': completed, 'plan_sha256': m.sha(m.HERE / 'plan.json'),
                                          'new_GPU_or_waiter': False, 'real_test_pixels': False})


if __name__ == '__main__': main()
