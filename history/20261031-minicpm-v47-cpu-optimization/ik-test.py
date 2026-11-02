import os,subprocess,pathlib,json
p=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-cpu-optimization');s=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4');env=os.environ.copy();env['GLIBC_TUNABLES']='glibc.malloc.hugetlb=1';order=[0,1,4,2,3,5,6,7]
for n in [16,32]:
 cpus=[48+8*j+i for i in order for j in range(6)][:n]; cpu=','.join(map(str,cpus)); name=f'ik-t{n}';cmd=['numactl','--physcpubind=48-95','--membind=1',str(p/'ik-build/bin/llama-bench'),'-m',str(s/'model.complete.gguf'),'-ngl','0','-mmp','0','-t',str(n),'-cr',cpu,'-p','0','-n','64','-r','2','-o','json'];(p/(name+'-command.json')).write_text(json.dumps(cmd,indent=2));print('START',n,flush=True)
 with (p/(name+'.json')).open('w') as out,(p/(name+'.log')).open('w') as err:r=subprocess.run(cmd,stdout=out,stderr=err,env=env)
 print('DONE',n,r.returncode,flush=True)
 if r.returncode: (p/'IK_LOAD_FAILED').write_text(str(r.returncode));break
else:(p/'IK_BENCH_FINISHED').write_text('complete\n')
