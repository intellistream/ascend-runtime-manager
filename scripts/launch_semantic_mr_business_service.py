#!/usr/bin/env python3
"""Start a task-specific process in the verified existing user's runtime."""
import argparse,hashlib,json,os,shlex,socket,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hust_ascend_manager.container import resolve_docker_command
NAME='vllm-hust-shuhao-21rc'
IMAGE='sha256:df36f0215ab68aca019f7a612214e43013363003a838afb57cdbe0d5f6df9e48'
TASK='/home/shuhao/semantic-mr-business-dev-20260908'
p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
if a.receipt.exists():raise SystemExit('receipt already exists')
docker=resolve_docker_command()
if not docker:raise SystemExit('existing Docker authorization required')
r=json.loads(subprocess.check_output(docker+['inspect',NAME],text=True))[0]
if r['Image']!=IMAGE or not r['State']['Running']:raise SystemExit('runtime identity mismatch')
def keys(text):
 result=set()
 for line in text.splitlines():
  fields=line.split()
  for i,k in enumerate(fields[:-1]):
   if k.startswith(('ssh-','ecdsa-')):result.add(hashlib.sha256((k+' '+fields[i+1]).encode()).hexdigest());break
 return result
runtime_keys=keys(subprocess.check_output(docker+['exec','--user','0:0',NAME,'cat','/opt/bootstrap/authorized_keys'],text=True))
match=runtime_keys&keys(Path('/home/shuhao/.ssh/authorized_keys').read_text())
if not match:raise SystemExit('runtime owner key does not match current user access')
usage=subprocess.check_output(['npu-smi','info'],text=True)
if 'No running processes found in NPU 5 ' not in usage:raise SystemExit('physical NPU5 is occupied')
with socket.socket() as sock:sock.bind(('127.0.0.1',19495))
service=TASK+'/service';model='/home/shuhao/semantic-mr-pilot-20260908/Qwen2.5-7B-Instruct'
subprocess.run(docker+['exec','--user','0:0',NAME,'mkdir','-p',TASK],check=True)
subprocess.run(docker+['exec',NAME,'python3','-c',"import socket; s=socket.socket(); s.bind(('127.0.0.1',19495)); s.close()"],check=True)
subprocess.run(docker+['exec','--user','0:0',NAME,'mkdir',service],check=True)
args=['vllm','serve',model,'--served-model-name','Qwen2.5-7B-Instruct','--host','127.0.0.1','--port','19495',
 '--tensor-parallel-size','1','--dtype','bfloat16','--max-model-len','16384','--max-num-seqs','1',
 '--max-num-batched-tokens','16384','--gpu-memory-utilization','0.6','--no-enable-prefix-caching']
cmd=docker+['exec','-d','--user','0:0','--workdir',service]
for key,value in {'ASCEND_RT_VISIBLE_DEVICES':'5','TORCH_DEVICE_BACKEND_AUTOLOAD':'0','PYTHONNOUSERSITE':'1',
 'VLLM_CACHE_ROOT':service+'/vllm-cache','HF_HOME':service+'/hf-cache','XDG_CACHE_HOME':service+'/cache',
 'XDG_CONFIG_HOME':service+'/config','TRITON_CACHE_DIR':service+'/triton-cache','ASCEND_WORK_PATH':service+'/ascend',
 'TORCHINDUCTOR_CACHE_DIR':service+'/inductor'}.items():cmd+=['-e',key+'='+value]
runner="import os; from pathlib import Path; Path("+repr(service+'/server.pid')+").write_text(str(os.getpid())); os.execvp('vllm',"+repr(args)+")"
cmd+=[NAME,'/bin/sh','-c','exec python3 -c '+shlex.quote(runner)+' > '+shlex.quote(service+'/server.log')+' 2>&1']
receipt={'container_id':r['Id'],'image_id':r['Image'],'container_name':NAME,'owner_key_matches':sorted(match),
 'uid':0,'physical_npu':5,'command':cmd,'server_args':args,'prelaunch_npu':usage,
 'mode':'normal supported compilation/prefill defaults; prefix cache disabled to prevent cross-arm prompt reuse',
 'scope':'task-specific service process, no existing process or installation replaced'}
a.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
subprocess.run(cmd,check=True)
print(json.dumps({'container_id':r['Id'],'server_log':service+'/server.log','port':19495}))
