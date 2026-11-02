import os,subprocess,pathlib,time,json
p=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-cpu-optimization');s=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4');env=os.environ.copy();env['GLIBC_TUNABLES']='glibc.malloc.hugetlb=1'
cpus=[48,49,52,56,57,60,64,65,68,72,73,76,80,81,88,89]; cpu=','.join(map(str,cpus));mask=hex(sum(1<<i for i in cpus))
events='cycles,instructions,cache-misses,ls_refills_from_sys.ls_mabresp_lcl_dram,ls_hw_pf_dc_fill.ls_mabresp_lcl_dram,ls_refills_from_sys.ls_mabresp_rmt_dram'
for depth in [2800,9475]:
 name=f'counters-depth-{depth}';cmd=['numactl','--physcpubind=48-95','--membind=1',str(s/'build/bin/llama-bench'),'-m',str(s/'model.complete.gguf'),'-ngl','0','-lm','none','-t','16','-C',mask,'--cpu-strict','1','-p','0','-n','512','-d',str(depth),'-r','1','-o','jsonl','--progress']; (p/(name+'-command.json')).write_text(json.dumps(cmd,indent=2));print('START',name,flush=True)
 with (p/(name+'.jsonl')).open('w') as out,(p/(name+'.log')).open('w') as err:
  process=subprocess.Popen(cmd,stdout=out,stderr=err,env=env)
  while process.poll() is None:
   if 'generation run 1/1' in (p/(name+'.log')).read_text():break
   time.sleep(.1)
  if process.poll() is not None:raise RuntimeError('benchmark failed before generation')
  with (p/(name+'-perf.txt')).open('w') as perf: rc=subprocess.run(['sudo','-n','perf','stat','-x,','-a','-C',cpu,'-e',events,'-M','nps1_die_to_dram','--','sleep','15'],stdout=perf,stderr=subprocess.STDOUT).returncode
  if rc:process.terminate();raise RuntimeError('perf failed')
  if process.wait():raise RuntimeError('bench failed')
 print('DONE',name,flush=True)
(p/'COUNTERS_FINISHED').write_text('complete\n')
