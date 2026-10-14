from pathlib import Path
import sys,ast,copy
H=Path(__file__).resolve().parent;R=H.parent
sys.path.insert(0,str(R/'80_pgsnet_module_qualification'));import qualify80 as q
prior=R/'86_controlled_inference_preparation';old=q.read(prior/'contract_v2.json');q.verify(old);q.guard.require_idle();resources=q.resources(17)
assert q.a.digest(prior/'artifact_manifest.json')=='53e45e80857c0f4cbf3254f5c2a26afe35cfab4198eef783b6dc53c0853f3634'
assert q.read(prior/'independent_assessment.json')['passed']
protocol=q.read(prior/'prospective_protocol.json');assert not protocol['measurement_execution_allowed']
runs=[r for r in q.read(R/'73_device_specific_evaluation/contract_v2.json')['runs'] if r['dataset']=='PCOD' and r['entry']['config']['method'] in protocol['methods']]
assert len(runs)==25 and {r['entry']['config']['run_id'] for r in runs}==set(protocol['run_order'])
qualification=[next(r['entry']['config']['run_id'] for r in runs if r['entry']['config']['method']==m and r['entry']['config']['seed']==42) for m in protocol['methods']]
sources=dict(old['sources'])
for rel in ['artifact_manifest.json','contract_v2.json','independent_assessment.json','prospective_protocol.json','pilot_v2/complete.json','meta_v2/complete.json']:
 p=prior/rel;sources[str(p)]=q.a.digest(p)
for r in runs:
 for name in ['config.json','identity.json','portable_complete.json']:
  p=Path(r['entry']['config']['output_dir'])/name;sources[str(p)]=q.a.digest(p)
for p in H.glob('*.py'):
 ast.parse(p.read_text(encoding='utf-8-sig'));sources[str(p)]=q.a.digest(p)
c={k:old[k] for k in ['locks','runtime','expected_worker_image','venv_prefix','expected_torch','expected_directml','device_index','expected_device_name','launch_available_ram_gib','minimum_available_ram_gib','abort_available_ram_gib','abort_available_commit_gib','minimum_disk_gib','private_process_cap_gib','working_set_cap_gib','job_commit_cap_gib','job_active_process_limit']}
c.update(sources=sources,runs=runs,protocol=protocol,qualification_order=qualification,allow_matrix_only_after_all_five_qualified=True,qualification_forwards=35,per_worker_timeout_seconds=180,input_sha256='9206cd76f5af2941ecead46b05d900f36e57192645e016376cafb5c86c60981d',parameter_counts={x['method']:x['trainable_parameters'] for x in q.read(R/'83_capacity_compute_evidence/analysis_v1/compute_summary.json')},training_allowed=False,numerical_comparison_allowed=False,PGS_allowed=False,resource_budgets_increased=False,purpose='Prospective completed-output timing. Phase1 five seed42 resource qualifications, phase2 full25 only after independently audited phase1. Qualification timings retained but excluded from paper matrix. CPU safe57 original-state loading before device construction; source values verified before and after CPU load. No formal test images.',source_protocol_unchanged=True)
q.verify(c);q.save(H/'contract.json',c);q.save(H/'preflight.json',{'contract_sha256':q.a.digest(H/'contract.json'),'sources':len(sources),'resources':resources,'utc':q.utc()});print({'sources':len(sources),'contract_sha256':q.a.digest(H/'contract.json'),'resources':resources})
