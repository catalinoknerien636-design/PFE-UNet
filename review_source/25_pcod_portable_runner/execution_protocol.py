"""Fail-closed admission for a not-yet-frozen formal PCOD queue."""
import argparse
import json
from pathlib import Path

import portable_core as core

HERE, ROOT, t = core.HERE, core.ROOT, core.t
PROTOCOL = HERE / 'formal_protocol.json'
REVIEW = HERE / 'prelaunch_review.json'
TESTS = HERE / 'cpu_test_results.json'
RULES = {
    'initialization': 'from_scratch_no_historical_checkpoint_warm_start',
    'selection': 'maximum_pooled_validation_iou_earliest_exact_tie',
    'test_evaluation': False,
    'statistical_unit': 'matched_seed_labels_not_images_or_directions',
    'sampling': 'legacy_global_torch_rng_not_identical_batch_order_across_architectures',
    'scope': 'official_record_group_protocol_not_physical_scene_independent',
    'historical_backend_mixing': False,
    'additional_scene_and_registration_work_omitted': False,
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def hardware_report(report, state, expected_methods, protocol_sha):
    if (report.get('passed') is not True or report.get('protocol_sha256') != protocol_sha
            or state.get('state') != 'completed' or state.get('protocol_sha256') != protocol_sha
            or report.get('formal_experiments_completed') != 0 or report.get('test_evaluated') is not False):
        raise ValueError('Hardware acceptance not completed under the registered protocol')
    for key in ('backend_parity', 'updates', 'gpu_resume'):
        rows = report.get(key, [])
        if len(rows) != len(expected_methods) or {r['method'] for r in rows} != set(expected_methods):
            raise ValueError('Missing or duplicated hardware method')
    if not all(r.get('passed') is True and r.get('atol') == 1e-6 and r.get('rtol') == 1e-4
               for r in report['gpu_resume']):
        raise ValueError('Resume acceptance missing or tolerance changed')
    if not all(r.get('optimizer_steps') == 16 and r.get('finite') is True
               and r.get('dolp_input_gradient_l1_after_updates_cpu_check', 0) > 0 for r in report['updates']):
        raise ValueError('Full-resolution update or auxiliary dependency check missing')
    if not all(r.get('atol') == 1e-5 and r.get('rtol') == 1e-4 for r in report['backend_parity']):
        raise ValueError('Backend parity tolerance changed')


def hardware_evidence():
    supplemental = core.acceptance
    # Require the supplemental artifacts before expensive checks; never substitute
    # CPU tests or the four-method report for these two device checks.
    if not supplemental.REPORT.is_file() or not supplemental.STATUS.is_file():
        raise ValueError('Direct/learned-fusion device acceptance is still pending')
    supplemental.validate_protocol(read(supplemental.PROTOCOL))
    supplemental.verify_development(read(supplemental.PROTOCOL))
    old = core.backend
    hardware_report(read(old.REPORT), read(old.STATUS), old.h.METHODS, t.sha(old.PROTOCOL))
    hardware_report(read(supplemental.REPORT), read(supplemental.STATUS),
                    supplemental.METHODS, t.sha(supplemental.PROTOCOL))
    dependency = read(supplemental.HERE / 'dependency_status.json')
    if dependency.get('state') != 'completed' or dependency.get('protocol_sha256') != t.sha(supplemental.PROTOCOL):
        raise ValueError('Supplemental controller has not completed successfully')
    paths = [old.PROTOCOL, old.REPORT, supplemental.PROTOCOL, supplemental.REPORT,
             supplemental.HERE / 'dependency_status.json']
    return {str(path.relative_to(ROOT)): t.sha(path) for path in paths}


def review_evidence():
    review = read(REVIEW)
    expected = {'hardware_reviewed': True, 'pcod_scene_association_limit_reviewed': True,
                'official_results_not_scene_independence_claim': True,
                'new_scene_protocol_still_required': True, 'test_used_for_design': False,
                'budget_epochs': 50, 'seed_order': [42, 2026, 3407, 7819, 12011],
                'methods': list(core.candidate.METHODS)}
    if review.get('decision') != 'approve_corrected_official_protocol_only' or review.get('checks') != expected:
        raise ValueError('Common scientific protocol review is incomplete')
    evidence = review.get('evidence', {})
    required = {'02_lightweight_validation/split_audit.json',
                '20_pcod_backend_diagnostics/PROTOCOL_MIGRATION_PLAN_ZH.md',
                '07_reports/RESEARCH_REVISION_REPORT_ZH.md'}
    if not required <= evidence.keys() or not review.get('rationale', '').strip():
        raise ValueError('Review lacks traceable input/scene/numerical evidence')
    for name, expected_sha in evidence.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or t.sha(path) != expected_sha:
            raise ValueError('Review evidence path/hash differs')
    tests = read(TESTS)
    if tests.get('passed') is not True or tests.get('gpu_started') is not False:
        raise ValueError('Portable runtime CPU validation not passed')
    expected_sources = {str(p.relative_to(ROOT)): t.sha(p)
                        for p in (*core.RUNTIME_FILES, HERE / 'test_portable.py')}
    if tests.get('source_sha256') != expected_sources:
        raise ValueError('CPU tests were run against different runtime sources')
    return {'review_sha256': t.sha(REVIEW), 'cpu_tests_sha256': t.sha(TESTS)}


def specification(enabled):
    configs = core.configurations(enabled)
    items = []
    for cfg in configs:
        prospective = dict(cfg, training_authorized=True, protocol_frozen=True, candidate_only=False)
        items.append({'config': cfg, 'identity': core.identity(prospective)})
    return {'queue_id': 'pcod_corrected_backend_full_reference_v2',
            'execution_enabled': enabled, 'formal_protocol_frozen': enabled,
            'rules': RULES, 'configs': items,
            'candidate_matrix_sha256': t.sha(core.candidate.HERE / 'candidate_matrix.json'),
            'source_sha256': {str(p.relative_to(ROOT)): t.sha(p) for p in core.RUNTIME_FILES}}


def validate(protocol):
    if protocol.get('execution_enabled') is not True or protocol.get('formal_protocol_frozen') is not True:
        raise ValueError('Formal execution is disabled')
    expected = specification(True)
    expected['hardware_evidence'] = hardware_evidence()
    expected['review_evidence'] = review_evidence()
    if protocol != expected:
        raise ValueError('Frozen common protocol, scope, configuration or evidence changed')
    return protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare-disabled', 'freeze', 'validate'))
    args = parser.parse_args()
    if args.action == 'prepare-disabled':
        value = specification(False)
        with (HERE / 'disabled_protocol_proposal.json').open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2)
        print(json.dumps({'prepared_configs': len(value['configs']), 'execution_enabled': False}))
    elif args.action == 'freeze':
        freeze()
    else:
        validate(read(PROTOCOL))


def freeze():
    if PROTOCOL.exists():
        raise FileExistsError('Formal protocol already exists; no overwrite')
    # Resolve every gate before creating the release, so a failed gate leaves
    # neither an empty protocol nor partly activated configuration files.
    hardware = hardware_evidence()
    review = review_evidence()
    proposal = read(HERE / 'disabled_protocol_proposal.json')
    if proposal != specification(False):
        raise ValueError('Disabled proposal no longer matches the reviewed runtime')
    value = specification(True)
    value['hardware_evidence'], value['review_evidence'] = hardware, review
    validate(value)
    with PROTOCOL.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
    print(json.dumps({'frozen_protocol_sha256': t.sha(PROTOCOL), 'gpu_started': False}))


if __name__ == '__main__':
    main()
