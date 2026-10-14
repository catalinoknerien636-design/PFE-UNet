"""Freeze metadata-only evaluation definition after CPU tests; never starts work."""
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import offline_evaluate as o
import run_checks

HERE=o.HERE


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    target=HERE/'evaluation_contract.json'
    if target.exists():raise FileExistsError('Definition already frozen')
    tests=o.read(HERE/'cpu_attempt_v1/results.json')
    if not tests['passed'] or tests['tests']!=40 or tests['source_sha256']!=run_checks.sources():
        raise ValueError('Expected complete, source-bound CPU checks')
    root=o.ROOT
    inputs=o.read(root/'40_rgbp_formal_preparation/input_contract.json')
    for parts in inputs['manifests'].values():
        for item in parts.values():
            if o.sha(item['path'])!=item['sha256']:raise ValueError('Existing manifest changed')
    a,b=(inputs['manifests'][name]['test'] for name in o.e.VARIANTS)
    assert a['sha256']==b['sha256']
    rows=csv_rows(a['path']);ids=[r['sample_id'] for r in rows]
    assert len(ids)==len(set(ids))==1304 and all(r['split']=='test' for r in rows)
    strata_path=root/'22_rgbp_scene_geometry/results_v1/test_association_strata.csv'
    strata=csv_rows(strata_path);strata={r['sample_id']:r['geometric_link_partition'] for r in strata}
    assert set(strata)==set(ids) and Counter(strata.values())=={'none_detected':849,'train':364,'validation':91}
    integrity_path=root/'10_rgbp_glass_preparation/audit/file_integrity.csv'
    integrity={r['path']:r for r in csv_rows(integrity_path)}
    members=[]
    for row in rows:
        expected='test/mask/'+row['sample_id']+'_mask.png'
        assert row['mask']==expected and integrity[expected]['modality']=='mask'
        members.append({'sample_id':row['sample_id'],'mask':expected,
                        'mask_sha256':integrity[expected]['sha256'],'stratum':strata[row['sample_id']]})
    sources=dict(tests['source_sha256'])
    paths=[HERE/'SCIENTIFIC_REVIEW_ZH.md',HERE/'cpu_attempt_v1/results.json',
        root/'06_experiment_preparation/EXPERIMENT_QUEUE.json',root/'41_rgbp_portable_runner/formal_protocol.json',
        root/'41_rgbp_portable_runner/prelaunch_review.json',strata_path,integrity_path,
        root/'22_rgbp_scene_geometry/results_v1/stratum_protocol.json',
        root/'40_rgbp_formal_preparation/input_contract.json',Path(a['path']),Path(b['path'])]
    sources.update({str(p.relative_to(root)):o.sha(p) for p in paths})
    contract={'utc':datetime.now(timezone.utc).isoformat(),'definition_frozen':True,
        'training_protocol_changed':False,'inference_started':False,'gpu_started':False,
        'formal_results_exist':False,'PGSNet_source_admitted':False,
        'common_grid':[352,352],'probability':'raw FP32 sigmoid, no PNG, no minmax',
        'PGSNet_technical_resize':'416 to352, bilinear half-pixel, probability before threshold',
        'threshold':'strictly >0.5','mask':'binary then OpenCV nearest to352',
        'dense_denominator_epsilon':1e-7,'mae':'CPU Torch FP32 per-image mean, equal-image mean',
        'boundary_primary':'image-mean Boundary F1 at2px','boundary_tolerances':[1,2,4],
        'boundary_band_ratio':.02,'boundary_both_empty':1,'dense_empty_union':0,
        'seeds':list(o.e.SEEDS),'statistical_unit':'matched training seed',
        'primary_comparisons':'PFE-minus-Direct IoU within each of two variants; Holm2 exploratory signflip',
        'secondary_analyses':'other methods/metrics/strata/boundaries descriptive',
        'sources_must_complete':'stage41 exact50 epochs, selected best validation checkpoint and all exports',
        'physical_scene_independence':False,'source_sha256':sources,'members':members}
    with target.open('x',encoding='utf-8') as f:json.dump(contract,f,indent=2,allow_nan=False)
    print(json.dumps({'definition_sha256':o.sha(target),'members':len(members),
                      'strata':dict(Counter(strata.values())),'test_pixels_read':False,'gpu_started':False}))


if __name__=='__main__':main()
