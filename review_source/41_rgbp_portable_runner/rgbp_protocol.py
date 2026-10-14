"""Explicit scientific admission and immutable RGBP release; freezing uses no GPU."""
import argparse
from pathlib import Path
import rgbp_core as core

HERE, ROOT, t = core.HERE, core.ROOT, core.t
read = core.read
PROTOCOL = HERE / 'formal_protocol.json'
REVIEW = HERE / 'prelaunch_review.json'
TESTS = HERE / 'cpu_test_results.json'
PROPOSAL = HERE / 'disabled_protocol_proposal.json'
TEST_FILES = (HERE / 'test_runner41.py', HERE / 'run_checks.py')
RULES = {
    'from_scratch':True, 'development_seed_used_as_formal':False, 'test_evaluation':False,
    'selection':'maximum_pooled_validation_iou_earliest_exact_tie',
    'training_repetition_unit':'three_seed_labels_per_method_within_protocol',
    'primary_comparison':'pfe_multiply_vs_direct_within_each_protocol',
    'secondary_comparisons':'rgb_only_learned_fusion_global_film_descriptive',
    'cross_protocol_change_is_causal_leakage_effect':False,
    'association_graph_is_physical_scene_metadata':False, 'E05_completed':False,
    'historical_backend_mixing':False,
}
REVIEW_CHECKS = {
    'decision':'approve_official_and_association_sensitivity_not_physical_scene_proof',
    'epochs':50, 'seeds':[42,2026,3407], 'methods':list(core.candidate.METHODS),
    'variants':list(core.candidate.VARIANTS), 'candidate_runs':30,
    'all_development_curves_reviewed':True, 'terminal_fusion_best_and_nonconvergence_disclosed':True,
    'association_additional_15_runs_and_cost_reviewed':True,
    'unequal_population_and_update_count_confounding_reviewed':True,
    'full_official_test_and_frozen_strata_preserved':True,
    'physical_scene_independence_still_unresolved':True, 'E05_not_replaced':True,
    'three_seed_exact_two_sided_minimum_p_is_0_25':True,
    'test_used_for_design':False, 'device_acceptance_reviewed':True,
}
REVIEW_REQUIRED = (
    '34_rgbp_protocol_review/verification.json',
    '34_rgbp_protocol_review/proposal_v1/proposal.json',
    '33_rgbp_finite_omission_audit/README_ZH.md',
    '39_rgbp_development_review/complete_development_review.json',
    '40_rgbp_formal_preparation/integration_verification.json',
)


def hardware_evidence():
    return core.candidate.development.verify_hardware()


def validate_review(record):
    if record.get('checks') != REVIEW_CHECKS or not record.get('rationale','').strip():
        raise ValueError('Scientific admission incomplete')
    evidence = record.get('evidence',{})
    if not set(REVIEW_REQUIRED) <= evidence.keys():
        raise ValueError('Scientific review source chain missing')
    for name, digest in evidence.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or t.sha(path) != digest:
            raise ValueError('Scientific review source path/hash differs')


def review_evidence():
    validate_review(read(REVIEW))
    tests = read(TESTS)
    expected = {str(p.relative_to(ROOT)):t.sha(p) for p in (*core.RUNTIME_FILES,*TEST_FILES)}
    if (tests.get('passed') is not True or tests.get('gpu_started') is not False
            or tests.get('source_sha256') != expected or tests.get('tests',0) < 20):
        raise ValueError('Runtime tests not bound to this release')
    return {'review_sha256':t.sha(REVIEW),'cpu_tests_sha256':t.sha(TESTS)}


def specification(enabled):
    configs = core.configurations(enabled)
    return {'queue_id':'rgbp_corrected_official_and_association_v1',
        'execution_enabled':enabled, 'formal_protocol_frozen':enabled, 'rules':RULES,
        'configs':[{'config':cfg, 'identity':core.identity(dict(cfg,training_authorized=True,
            protocol_frozen=True,candidate_only=False))} for cfg in configs],
        'preparation_evidence':core.verify_preparation(),
        'source_sha256':{str(p.relative_to(ROOT)):t.sha(p) for p in core.RUNTIME_FILES}}


def validate(value):
    if value.get('execution_enabled') is not True or value.get('formal_protocol_frozen') is not True:
        raise ValueError('No frozen formal execution authorization')
    hardware, review = hardware_evidence(), review_evidence()
    expected = specification(True)
    expected.update(hardware_evidence=hardware, review_evidence=review)
    if value != expected:
        raise ValueError('Frozen protocol, matrix, source or evidence differs')
    return value


def freeze():
    if PROTOCOL.exists():
        raise FileExistsError('Formal protocol already exists')
    hardware, review = hardware_evidence(), review_evidence()
    if read(PROPOSAL) != specification(False):
        raise ValueError('Disabled proposal changed')
    value = specification(True)
    value.update(hardware_evidence=hardware,review_evidence=review)
    validate(value)
    core.candidate.save_new(PROTOCOL,value)
    return {'protocol_sha256':t.sha(PROTOCOL),'gpu_started':False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare-disabled','freeze','validate'))
    args = parser.parse_args()
    if args.action == 'prepare-disabled':
        core.candidate.save_new(PROPOSAL,specification(False))
    elif args.action == 'freeze':
        print(freeze())
    else:
        validate(read(PROTOCOL))


if __name__ == '__main__':
    main()
