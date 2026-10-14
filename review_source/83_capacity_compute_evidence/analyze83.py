from pathlib import Path
import contextlib,csv,gc,json,math,msvcrt,os,statistics,sys,traceback
H=Path(__file__).resolve().parent;R=H.parent
sys.path.insert(0,str(R/'80_pgsnet_module_qualification'));import qualify80 as q
METHODS=['direct_reference','pfe_reference','learned_fusion_reference','global_film_style','wide_direct']
def csvrows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_csv(p,rows):
    with Path(p).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stats(xs):return {'mean':statistics.mean(xs),'sample_sd':statistics.stdev(xs),'minimum':min(xs),'maximum':max(xs)}

def analytical(method,b,size=352):
    # Independent architecture-level arithmetic, without reading observed hook rows.
    def dc(ci,co,h):return {'p':9*(ci*co+co*co)+4*co,'m':h*h*9*(ci*co+co*co)}
    p=0;m=0;channels=[b*2**i for i in range(5)]
    streams=[4] if method in ('direct_reference','wide_direct') else [3,1]
    for start in streams:
        ci=start
        for i,co in enumerate(channels):
            v=dc(ci,co,size//2**i);p+=v['p'];m+=v['m'];ci=co
    for i in range(3,-1,-1):
        co=channels[i];v=dc(channels[i+1]+co,co,size//2**i);p+=v['p'];m+=v['m']
    p+=b+1;m+=size*size*b
    if method=='pfe_reference':
        for i,co in enumerate(channels):
            mid=max(co//4,8);p+=2*co*mid+2*mid+co;m+=(size//2**i)**2*2*co*mid
    elif method=='learned_fusion_reference':
        for i,co in enumerate(channels):p+=2*co*co+2*co;m+=(size//2**i)**2*2*co*co
    elif method=='global_film_style':
        for co in channels:p+=2*co*co+2*co;m+=2*co*co
    return {'trainable_parameters':p,'conv_MACs':m}

class ConvCount:
    def __init__(self,torch):self.torch=torch;self.rows=[];self.leaf_calls=[]
    def hook(self,name):
        def capture(m,args,y):
            if isinstance(m,self.torch.nn.Conv2d):
                x=args[0];n,ci,hi,wi=map(int,x.shape);no,co,ho,wo=map(int,y.shape)
                assert n==no and ci==m.in_channels and co==m.out_channels
                expected=[(v+2*p-d*(k-1)-1)//s+1 for v,p,d,k,s in zip((hi,wi),m.padding,m.dilation,m.kernel_size,m.stride)]
                assert expected==[ho,wo]
                terms=(ci//m.groups)*math.prod(m.kernel_size);mac=n*co*ho*wo*terms
                self.rows.append({'module':name,'input_shape':list(x.shape),'output_shape':list(y.shape),'in_channels':ci,'out_channels':co,'kernel':list(m.kernel_size),'groups':m.groups,'MACs':mac,'bias_additions_excluded':int(y.numel()) if m.bias is not None else 0})
            self.leaf_calls.append({'module':name,'type':type(m).__name__})
            return None
        return capture
    def attach(self,model):return [m.register_forward_hook(self.hook(n)) for n,m in model.named_modules() if not list(m.children())]

def selftests(torch):
    checks=[]
    fixtures=[(torch.nn.Conv2d(4,6,3,padding=1,groups=2),[2,4,7,7],2*6*7*7*2*9),(torch.nn.Conv2d(3,3,3,padding=2,dilation=2,stride=2,groups=3),[1,3,7,7],1*3*4*4*9),(torch.nn.Conv2d(2,5,1),[1,2,3,3],1*5*3*3*2)]
    for i,(model,shape,expected) in enumerate(fixtures):
        x=torch.ones(shape);plain=model(x);meter=ConvCount(torch);handles=meter.attach(model);actual=model(x)
        assert torch.equal(actual,plain) and sum(r['MACs'] for r in meter.rows)==expected
        for h in handles:h.remove()
        checks.append('conv_formula_'+str(i))
    class Twice(torch.nn.Module):
        def __init__(self):super().__init__();self.conv=torch.nn.Conv2d(2,2,1,bias=False);self.unused=torch.nn.Conv2d(2,3,1)
        def forward(self,x):return self.conv(self.conv(x))
    model=Twice().to('meta');counter=ConvCount(torch);handles=counter.attach(model);y=model(torch.ones(1,2,4,4,device='meta'));y.sum().backward()
    assert sum(x['MACs'] for x in counter.rows)==128 and len(counter.rows)==2 and model.unused.weight.grad is None and model.conv.weight.grad is not None
    for h in handles:h.remove()
    checks.append('shared_call_count_and_unused_parameter_negative_control')
    return {'passed':True,'checks':checks}

def run():
    c=q.read(H/'contract.json');q.verify(c);q.guard.require_idle();q.resources(10)
    out=H/'analysis_v1';out.mkdir(exist_ok=False)
    with contextlib.ExitStack() as locks:
        for rel in c['locks']:
            f=locks.enter_context((R/rel).open('r+b'));f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        before=q.old.protection();q.save(out/'identity.json',{'owner':q.guard.process_identity(os.getpid()),'contract_sha256':q.a.digest(H/'contract.json'),'resources':q.resources(10),'utc':q.utc()})
        try:
            torch=q.a.require_runtime();torch.set_num_threads(1)
            q.save(out/'counter_tests.json',selftests(torch))
            sys.path.insert(0,str(R/'15_pcod_training'));import pcod_models as models
            sys.path.insert(0,str(R/'20_pcod_backend_diagnostics'));import resize_candidate as resize;import resize_factor2 as factor2
            resize.bilinear_half_pixel=factor2.bilinear_half_pixel
            reports=[]
            for method in METHODS:
                width=41 if method=='wide_direct' else 32
                with torch.device('meta'):model=resize.install_unet_candidate(models.build_model(method,base_channels=width,seed=42)).train()
                assert all(p.device.type=='meta' for p in model.parameters())
                params=sum(p.numel() for p in model.parameters() if p.requires_grad);counter=ConvCount(torch);handles=counter.attach(model)
                output=model(torch.empty(1,4,352,352,device='meta',dtype=torch.float32));assert tuple(output.shape)==(1,1,352,352)
                output.sum().backward();connected=sum(p.numel() for p in model.parameters() if p.requires_grad and p.grad is not None)
                missing=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None];assert not missing
                macs=sum(v['MACs'] for v in counter.rows);formula=analytical(method,width);assert params==formula['trainable_parameters'] and macs==formula['conv_MACs']
                rows={'method':method,'base_channels':width,'trainable_parameters':params,'graph_connected_parameters':connected,'conv_MACs_per_image':macs,'conv_GMACs':macs/1e9,'conv_MACs_times2_Goperations':2*macs/1e9,'conv_calls':len(counter.rows),'nonconv_leaf_types_excluded':sorted({v['type'] for v in counter.leaf_calls if v['type']!='Conv2d'}),'analytical_crosscheck':formula,'connectivity_not_nonzero_gradient_evidence':True}
                reports.append(rows);q.save(out/(method+'_detail.json'),{'summary':rows,'convolutions':counter.rows,'leaf_calls':counter.leaf_calls})
                for h in handles:h.remove()
                del model,output,counter;gc.collect();q.resources(8)
            q.save(out/'compute_summary.json',reports)
            timing=[]
            for run in c['runs']:
                cfg=run['config'];folder=Path(cfg['output_dir']);marker=q.read(folder/'portable_complete.json')
                assert q.a.digest(folder/'portable_complete.json')==run['completion_sha256']
                assert marker['run_id']==cfg['run_id'] and marker['protocol_sha256']==c['formal_protocol_sha256']
                assert q.read(folder/'config.json')==cfg and q.read(folder/'status.json')['completed_epoch']==50
                history=csvrows(folder/'history.csv');assert len(history)==50 and [int(x['epoch']) for x in history]==list(range(1,51))
                assert q.a.digest(folder/'history.csv')==marker['artifact_sha256']['history.csv']
                assert all(int(x['train_images'])==3104 and int(x['val_images'])==776 for x in history)
                training=[float(x['train_seconds']) for x in history];validation=[float(x['validation_seconds']) for x in history]
                assert all(math.isfinite(v) and v>0 for v in training+validation)
                timing.append({'method':cfg['method'],'seed':cfg['seed'],'run_id':cfg['run_id'],'train_seconds_sum':sum(training),'validation_seconds_sum':sum(validation),'loop_seconds_sum':sum(training)+sum(validation),'train_seconds_per_epoch_mean':statistics.mean(training),'validation_seconds_per_epoch_mean':statistics.mean(validation),'train_seconds_per_epoch_median':statistics.median(training),'validation_seconds_per_epoch_median':statistics.median(validation),'train_seconds_per_image':sum(training)/(50*3104),'validation_seconds_per_image':sum(validation)/(50*776),'history_sha256':q.a.digest(folder/'history.csv'),'completion_sha256':run['completion_sha256']})
            write_csv(out/'timing_per_run.csv',timing)
            performance=csvrows(R/'74_test_results_manuscript/group_summary.csv');table=[]
            for comp in reports:
                method=comp['method'];times=[v for v in timing if v['method']==method];assert [x['seed'] for x in times]==[42,2026,3407,7819,12011]
                score=next(x for x in performance if x['protocol']=='pcod_record_group' and x['method']==method)
                row={k:comp[k] for k in ['method','base_channels','trainable_parameters','graph_connected_parameters','conv_MACs_per_image','conv_GMACs']}
                row.update(test_iou_mean=float(score['iou_mean']),test_iou_sample_sd=float(score['iou_sample_sd']),n=5)
                for metric in ['train_seconds_sum','validation_seconds_sum','loop_seconds_sum']:
                    for key,value in stats([x[metric]/3600 for x in times]).items():row[metric.replace('_seconds_sum','_hours')+'_'+key]=value
                table.append(row)
            write_csv(out/'capacity_compute_timing.csv',table);q.save(out/'capacity_compute_timing.json',table)
            q.verify(c);after=q.old.protection();q.guard.require_idle()
            q.save(out/'complete.json',{'passed':True,'five_corrected_meta_graphs':True,'all_25_histories_1250_epochs_verified':True,'actual_model_numeric_forward':False,'optimizer_updates':0,'GPU_used':False,'pretrained_weights_or_real_data_read':False,'not_inference_latency_or_peak_memory_benchmark':True,'source_protection_before':before,'source_protection_after':after,'resources':q.resources(8),'utc':q.utc()})
            print(json.dumps(table),flush=True)
        except BaseException as e:q.save(out/'failure.json',{'error':str(e),'traceback':traceback.format_exc(),'utc':q.utc()});raise
if __name__=='__main__':run()
