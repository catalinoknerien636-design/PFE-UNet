from pathlib import Path
import sys,ast
H=Path(__file__).resolve().parent;R=H.parent
sys.path.insert(0,str(R/'80_pgsnet_module_qualification'));import qualify80 as q
c=q.read(H/'contract.json');q.verify(c)
c['sources'][str(H/'contract.json')]=q.a.digest(H/'contract.json')
for name in ['profile86_v2.py','worker86_v2.py','supervisor86_v2.py','run86_v2.py','amend86.py']:
 p=H/name;ast.parse(p.read_text(encoding='utf-8-sig'));c['sources'][str(p)]=q.a.digest(p)
c['meta_runtime']=str(R/'31_pgsnet_checkpoint_safety/venv_cpu/Scripts/python.exe')
c['meta_torch']='2.14.0+cpu'
c['development_amendment']='Meta v1 was launched with bundled controller Python lacking torch, stopped before model construction; v2 uses original independent reviewed CPU2.14 runtime and fresh output. Prior contract/code/failure retained. No scientific protocol change.'
q.save(H/'contract_v2.json',c);q.save(H/'preflight_v2.json',{'contract_sha256':q.a.digest(H/'contract_v2.json'),'sources':len(c['sources']),'meta_runtime':c['meta_runtime'],'resources':q.resources(17),'utc':q.utc()});print(q.a.digest(H/'contract_v2.json'))
