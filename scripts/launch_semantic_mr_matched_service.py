#!/usr/bin/env python3
"""Launch one isolated, digest-pinned local pilot service with existing Docker rights."""
import argparse,json,os,pwd,socket,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hust_ascend_manager.container import resolve_docker_command,discover_device_args
IMAGE='sha256:f4c89c293e076453e9eef9edb5fb9669740dccbd3c48619a9f976d775fc29b81'
p=argparse.ArgumentParser();p.add_argument('--device',type=int,choices=[4,5,6],required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=19491)
p.add_argument('--name',default='semantic-mr-matched-20260908');a=p.parse_args()
if Path('/etc/machine-id').read_text().strip()!='96c516c8dd3943699fdf4c1cb71629a3':raise SystemExit('host mismatch')
docker=resolve_docker_command()
if not docker:raise SystemExit('existing Docker authorization unavailable')
if subprocess.run(docker+['inspect',a.name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:raise SystemExit('name already exists')
identity=json.loads(subprocess.check_output(docker+['image','inspect',IMAGE],text=True))[0]
if identity['Id']!=IMAGE:raise SystemExit('image mismatch')
with socket.socket() as s:s.bind(('127.0.0.1',a.port))
usage=subprocess.check_output(['npu-smi','info'],text=True)
if f'No running processes found in NPU {a.device} ' not in usage:raise SystemExit('device has reported processes')
a.output.mkdir(parents=True,exist_ok=False,mode=0o700)
output=a.output.resolve();(output/'prelaunch-npu.txt').write_text(usage)
(output/'image-inspect.json').write_text(json.dumps(identity,indent=2)+'\n')
os.environ['HUST_ASCEND_CONTAINER_NPU_DEVICES']=str(a.device)
model=Path('/data/shared_models/Qwen2.5-7B-Instruct')
cmd=docker+['run','-d','--name',a.name,'--label','owner=shuhao','--label','experiment=semantic-mr-bounded-pilot',
 '--user',f'{os.getuid()}:{os.getgid()}','--network','host','--shm-size','8g',*discover_device_args(),
 '-v',f'{model}:{model}:ro','-v',f'{output}:/pilot-output','--workdir','/pilot-output']
for source in ['/usr/local/Ascend/driver/lib64','/usr/local/Ascend/driver/version.info','/usr/local/dcmi','/etc/ascend_install.info','/usr/local/sbin/npu-smi']:
 if Path(source).exists():cmd+=['-v',f'{source}:{source}:ro']
for value in [f'LOGNAME={pwd.getpwuid(os.getuid()).pw_name}',f'USER={pwd.getpwuid(os.getuid()).pw_name}',f'ASCEND_RT_VISIBLE_DEVICES={a.device}','TORCH_DEVICE_BACKEND_AUTOLOAD=0','VLLM_CACHE_ROOT=/pilot-output/vllm-cache',
 'HF_HOME=/pilot-output/hf-cache','XDG_CACHE_HOME=/pilot-output/cache','TORCHINDUCTOR_CACHE_DIR=/pilot-output/inductor',
 'ASCEND_WORK_PATH=/pilot-output/ascend','PYTHONNOUSERSITE=1']:
 cmd+=['-e',value]
cmd+=['--entrypoint','vllm',IMAGE,'serve',str(model),'--served-model-name','Qwen2.5-7B-Instruct',
 '--host','127.0.0.1','--port',str(a.port),'--tensor-parallel-size','1','--dtype','bfloat16',
 '--max-model-len','16384','--max-num-seqs','1','--max-num-batched-tokens','16384',
 '--gpu-memory-utilization','0.6','--enforce-eager','--no-enable-prefix-caching','--no-enable-chunked-prefill']
(output/'launch.json').write_text(json.dumps({'command':cmd,'script':str(Path(__file__).resolve()),'image_id':IMAGE,
 'model':str(model),'physical_npu':a.device,'uid':os.getuid(),'port':a.port,'scope':'owned service startup; no pilot inference'},indent=2)+'\n')
result=subprocess.run(cmd,capture_output=True,text=True)
(output/'docker-run.json').write_text(json.dumps({'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr},indent=2)+'\n')
print(result.stdout or result.stderr,end='');raise SystemExit(result.returncode)
