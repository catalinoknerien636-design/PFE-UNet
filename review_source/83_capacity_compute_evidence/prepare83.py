from pathlib import Path
import ast,json,sys
H=Path(__file__).resolve().parent;R=H.parent
sys.path.insert(0,str(R/'80_pgsnet_module_qualification'));import qualify80 as q
ast.parse((H/'analyze83.py').read_text(encoding='utf-8-sig'))
p73=R/'73_device_specific_evaluation/contract_v2.json';assert q.a.digest(p73)=='1b93b1d2edb9a8a2a9a6f9b71d7f5d919e7684fd2253f653bcfc2f0c9d04d35e'
runs=[{'config':r['entry']['config'],'completion_sha256':r['completion_sha256']} for r in q.read(p73)['runs'] if r['dataset']=='PCOD' and r['entry']['config']['method']!='deeplabv3plus_resnet50'];assert len(runs)==25
sources={}
for rel in ['25_pcod_portable_runner/formal_protocol.json','25_pcod_portable_runner/portable_core.py','15_pcod_training/train_pcod.py','15_pcod_training/pcod_models.py','20_pcod_backend_diagnostics/resize_candidate.py','20_pcod_backend_diagnostics/resize_factor2.py','14_capacity_control/preflight.json','74_test_results_manuscript/group_summary.csv','74_test_results_manuscript/final_artifact_manifest.json','73_device_specific_evaluation/contract_v2.json','77_physics_availability_audit/manuscript/PFE_UNet_Manuscript_Evidence_Checked_v13.docx','74_test_results_manuscript/manuscript/PFE_UNet_Supplementary_With_Corrected_Tests_v8.docx','77_physics_availability_audit/artifact_manifest.json','80_pgsnet_module_qualification/qualify80.py','75_pgsnet_resource_profile/profile75_v2.py','36_pgsnet_pretrained_contract/adapter.py','24_pcod_reference_acceptance/process_guard.py','13_registration_evaluation/reference/04_multiscale_ablation/code/models/unet.py','13_registration_evaluation/reference/04_multiscale_ablation/code/models/pfe_unet_reference.py']:
    p=R/rel;sources[str(p)]=q.a.digest(p)
for p in H.glob('*.py'):sources[str(p)]=q.a.digest(p)
for run in runs:
    folder=Path(run['config']['output_dir']);p=folder/'portable_complete.json';assert q.a.digest(p)==run['completion_sha256'];m=q.read(p)
    for name in ['history.csv','config.json','status.json','identity.json','portable_complete.json']:
        f=folder/name;sources[str(f)]=q.a.digest(f)
    assert sources[str(folder/'history.csv')]==m['artifact_sha256']['history.csv']
    for p,h in m['identity']['source_sha256'].items():
        if str(R/p) in sources:assert sources[str(R/p)]==h
c={'purpose':'post-hoc complete five-U-Net capacity, convolution MAC, and historical loop-cost description; no new training/evaluation or tuning',
'sources':sources,'runs':runs,'formal_protocol_sha256':q.a.digest(R/'25_pcod_portable_runner/formal_protocol.json'),'locks':q.read(R/'79_pgsnet_checkpoint_candidate/contract.json')['locks'],
'scope':'all five PCOD U-Net methods and all5 registered seeds; ResNet50 separate backbone not in this capacity subanalysis; RGBP and PGS outside scope',
'input':[1,4,352,352],'dtype':'float32','model_execution':'meta shapes and autograd connection only; no learned checkpoint read or numerical model evaluation',
'MAC_definition':'N*Cout*Hout*Wout*(Cin/groups)*kh*kw summed over Conv2d invocations, dense zero-padding convention; per image1 MAC unit for each multiply-accumulate pair',
'exclusions':['bias additions','batch normalization','activations/sigmoid','pooling','pointwise gates/FiLM arithmetic','custom interpolation','data movement and framework','backward/optimizer','backend-specific convolution algorithms'],
'connected_parameters':'has meta autograd gradient from sum of output; shape-graph connection only, not measured nonzero or learned effective degrees of freedom',
'cost_definition':'each of25 original history CSVs has50 epochs; sum train_seconds and validation_seconds independently; mean/sample SD/min/max across five seeds; no dropped warmup/outlier epochs; no p-values for cost',
'cost_limitations':'run_epoch timer includes loader/transfers/loss/progress/optimization or validation metrics; excludes checkpoint writes after run_epoch and inter-run setup; no explicit final device synchronization, so not exact GPU kernel time or standalone inference latency; uncontrolled historical host load',
'original_results_unchanged':True,'GPU_allowed':False,'numerical_diagnosis_allowed':False,'minimum_available_ram_gib':8,'minimum_disk_gib':20,
'web_basis':[{'url':'https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html','supports':'grouped 2D convolution dimensions and channel connections'},{'url':'https://docs.pytorch.org/docs/2.14/profiler.html','supports':'operator FLOP counts are formula estimates for supported operators, not timing'}]}
q.verify(c);q.guard.require_idle();q.save(H/'contract.json',c);q.save(H/'preflight.json',{'resources':q.resources(10),'sources':len(sources),'contract_sha256':q.a.digest(H/'contract.json'),'utc':q.utc()})
print(json.dumps({'sources':len(sources),'runs':len(runs),'contract_sha256':q.a.digest(H/'contract.json')}))
