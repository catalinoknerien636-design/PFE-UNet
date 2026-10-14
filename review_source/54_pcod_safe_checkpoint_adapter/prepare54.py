"""Bind three already-verified completed local best files without unpickling."""
from pathlib import Path
import sys
from datetime import datetime, timezone
import metadata54 as m


def main():
    sys.path.insert(0, str(m.ROOT / '25_pcod_portable_runner'))
    import execution_protocol as protocol
    formal = protocol.validate(protocol.read(protocol.PROTOCOL))
    prior_path = m.ROOT / '53_pcod_formal_evaluation/integration_verification.json'
    prior = m.read(prior_path)
    m.need(prior['passed'] and len(prior['pcod_completed_verified']) == 3, 'Prespecified completed prefix evidence')
    m.need(prior['pcod_protocol_sha256'] == m.sha(protocol.PROTOCOL), 'Source formal protocol differs')
    checkpoints = []
    for evidence in prior['pcod_completed_verified']:
        item = next(x for x in formal['configs'] if x['config']['run_id'] == evidence['run_id'])
        cfg, identity = item['config'], item['identity']; out = Path(cfg['output_dir'])
        marker = m.read(out / 'portable_complete.json'); status = m.read(out / 'status.json')
        m.need(m.sha(out / 'portable_complete.json') == evidence['marker_sha256'], 'Prior completion changed')
        m.need(marker['identity'] == identity and marker['artifact_sha256'] == evidence['artifacts_sha256'], 'Prior identity/artifact binding changed')
        for name, digest in marker['artifact_sha256'].items(): m.need(m.sha(out / name) == digest, 'Original completed artifact changed')
        m.need(status['state'] == 'completed' and status['completed_epoch'] == 50 and status['test_evaluated'] is False, 'Full budget metadata required')
        checkpoints.append({'run_id': cfg['run_id'], 'identity': identity, 'output_dir': str(out),
            'completion_sha256': evidence['marker_sha256'], 'artifacts_sha256': evidence['artifacts_sha256'],
            'best_epoch': status['best_epoch'], 'best_validation_iou': status['best_validation_iou']})
    files = list(m.HERE.glob('*.py')) + [m.HERE / 'README.md', prior_path,
        m.ROOT / '53_pcod_formal_evaluation/artifact_manifest.json', m.ROOT / '53_pcod_formal_evaluation/evaluation_contract.json',
        m.ROOT / '52_pgsnet_streamed_checkpoint/tensor_store.py', protocol.PROTOCOL,
        m.ROOT / '25_pcod_portable_runner/prelaunch_review.json',
        m.ROOT / '31_pgsnet_checkpoint_safety/guarded_inspection.py', m.ROOT / '31_pgsnet_checkpoint_safety/prepare_runtime.py',
        m.ROOT / '31_pgsnet_checkpoint_safety/runtime_ready.json',
        m.ROOT / '31_pgsnet_checkpoint_safety/venv_cpu/Lib/site-packages/torch/torch_version.py',
        m.ROOT / '31_pgsnet_checkpoint_safety/venv_cpu/Lib/site-packages/torch/_weights_only_unpickler.py',
        Path('<LOCAL_HOST_PATH_04>')]
    plan = {'utc': datetime.now(timezone.utc).isoformat(), 'purpose': 'Owned completed PCOD best metadata normalization, not generic external weight conversion',
        'formal_protocol_sha256': m.sha(protocol.PROTOCOL), 'checkpoints': checkpoints,
        'source_sha256': {str(p): m.sha(p) for p in files}, 'source_selection': 'exact completed prefix already verified in stage53, not performance selection',
        'external_checkpoint_sources_allowed': False, 'original_files_modified': False,
        'restricted_load_runtime': '31_pgsnet_checkpoint_safety/venv_cpu', 'pickle_protocol': 2,
        'change': 'Only remove one unused TorchVersion class/one-tuple construction, preserve version literal and other bytes',
        'all_non_metadata_zip_members_must_match': True, 'default_weights_only_no_user_allowlist': True,
        'model_creation_or_GPU_or_test_pixels': False, 'stage53_is_not_edited_or_unblocked': True,
        'future_full_device_export_acceptance': False}
    m.write(m.HERE / 'plan.json', plan)
    print({'checkpoints': len(checkpoints), 'plan_sha256': m.sha(m.HERE / 'plan.json')})


if __name__ == '__main__': main()
