"""Outcome-blind candidate manifests and disabled budget50 matrix; no GPU."""
from collections import Counter
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import unittest
import formal_contract as f


def write_csv_new(path, rows):
    with path.open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    if (f.HERE/'input_contract.json').exists() or (f.HERE/'manifests_candidate').exists():
        raise RuntimeError('Preparation already attempted; preserve and inspect')
    source=f.ROOT/'34_rgbp_protocol_review'
    verification=f.read(source/'verification.json')
    assert verification['passed'] and verification['tests']==18
    for name,digest in verification['sources_sha256'].items():
        assert f.t.sha(f.ROOT/name)==digest,name
    for name,digest in verification['outputs_sha256'].items():
        assert f.t.sha(source/'proposal_v1'/name)==digest,name
    curves=f.read(f.ROOT/'39_rgbp_development_review/complete_development_review.json')
    assert curves['all_five_full_curves_reviewed'] and curves['tests']['passed']
    assert curves['budget_review']['recommended_common_epochs']==50 and not curves['test_used']
    for name,digest in curves['evidence_sha256'].items():
        assert f.t.sha(f.ROOT/name)==digest,name
    f.development.validate_queue(f.read(f.development.QUEUE))
    hardware=f.development.verify_hardware()
    original={p:f.csv_rows(f.ROOT/'10_rgbp_glass_preparation/candidate_split'/f'{p}.csv')
        for p in ('train','validation','test')}
    official={r['sample_id']:p for p,rows in original.items() for r in rows}
    assert len(official)==4510 and sum(map(len,original.values()))==4510
    members=f.csv_rows(source/'proposal_v1/membership_REVIEW_ONLY_NOT_TRAINING.csv')
    assert len(members)==4510
    assert all(r['training_authorized']=='False' and r['physical_scene_status']=='unresolved' for r in members)
    assert {r['sample_id']:r['original_partition'] for r in members}==official
    projected={r['sample_id']:r['proposed_partition_NOT_ACTIVE'] for r in members}
    f.validate_members(official,projected)
    assert Counter(projected.values())==Counter(train=1857,validation=456,test=1304,quarantine=893)
    manifests={}
    out=f.HERE/'manifests_candidate';out.mkdir()
    for variant in f.VARIANTS:
        folder=out/variant;folder.mkdir();manifests[variant]={}
        for part in ('train','validation','test'):
            rows=original[part] if variant==f.VARIANTS[0] else [r for r in original[part] if projected[r['sample_id']]==part]
            expected=1304 if part=='test' else f.COUNTS[variant][part]
            assert len(rows)==expected
            path=folder/f'{part}.csv';write_csv_new(path,rows)
            assert f.csv_rows(path)==rows
            manifests[variant][part]={'path':str(path),'sha256':f.t.sha(path),'count':len(rows)}
    assert manifests[f.VARIANTS[0]]['test']['sha256']==manifests[f.VARIANTS[1]]['test']['sha256']
    sources=[source/'verification.json',source/'proposal_v1/proposal.json',
        source/'proposal_v1/membership_REVIEW_ONLY_NOT_TRAINING.csv',
        f.ROOT/'39_rgbp_development_review/complete_development_review.json',f.development.QUEUE,
        f.ROOT/'21_portable_rgbp_development/configs/rgbp_dev_v3_half_pixel_direct_seed17.json']
    sources += [f.ROOT/'10_rgbp_glass_preparation/candidate_split'/f'{p}.csv' for p in original]
    inputs={'utc':datetime.now(timezone.utc).isoformat(),'training_authorized':False,
        'manifests':manifests,'source_sha256':{str(p.relative_to(f.ROOT)):f.t.sha(p) for p in sources},
        'original_rows_and_order_preserved':True,'new_images_written':False,'test_images_read':False,
        'physical_scene_status':'unresolved','quarantine_count_not_confirmed_leakage':893,
        'formal_scientific_approval':'pending','runtime_admission_and_gpu_mutex':'not implemented in this stage',
        'hardware_evidence_inherited_not_rerun':hardware}
    f.save_new(f.HERE/'input_contract.json',inputs)
    configs=f.candidates();identities=f.validate_matrix(configs)
    proposal={'utc':datetime.now(timezone.utc).isoformat(),'execution_enabled':False,
        'formal_protocol_frozen':False,'scientific_admission':False,'gpu_waiter_created':False,
        'purpose':'formal_preparation_not_a_release','configs':[
            {'config':cfg,'candidate_identity':identity} for cfg,identity in zip(configs,identities)],
        'source_sha256':{str(p.relative_to(f.ROOT)):f.t.sha(p) for p in f.SOURCES}}
    f.save_new(f.HERE/'disabled_budget50_candidates.json',proposal)
    log=io.StringIO()
    tests=unittest.TextTestRunner(stream=log,verbosity=2).run(
        unittest.defaultTestLoader.discover(str(f.HERE),pattern='test_formal40.py'))
    result={'utc':datetime.now(timezone.utc).isoformat(),'passed':tests.wasSuccessful(),
        'tests':tests.testsRun,'log':log.getvalue(),'gpu_started':False,'formal_training_started':False,
        'proposal_sha256':f.t.sha(f.HERE/'disabled_budget50_candidates.json'),
        'input_contract_sha256':f.t.sha(f.HERE/'input_contract.json'),
        'source_sha256':proposal['source_sha256']}
    f.save_new(f.HERE/'cpu_verification.json',result)
    if not tests.wasSuccessful():
        raise RuntimeError(log.getvalue())
    assert f.validate_matrix(configs)==identities
    print(json.dumps({'tests':tests.testsRun,'configs':len(configs),'counts':f.COUNTS,
        'passed':True,'execution_enabled':False,'proposal_sha256':result['proposal_sha256']},indent=2))


if __name__=='__main__':main()
