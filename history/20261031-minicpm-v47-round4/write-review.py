import pathlib,json,statistics
p=pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4'); baseline=pathlib.Path('/home/ubuntu/llm-evaluations/20261029-tiny-log-intelligence-round4/raw/MiniCPM5-2B')
incident=['network','disk_io','app_crash','gpu_driver','out_of_memory','auth_failures']
notes={
'small-1234':('Calls GPU hardware fault and storage hardware/driver cause confirmed; asserts image processing halted, data integrity risk, and underlying timeout recovery without evidence.','Treats retries as minor observation, not a separate major incident.'),
'small-2345':('Calls hardware failure and code bug confirmed; asserts GPU device recovered and implies memory limits without supporting evidence.','Treats recurring successful retries as performance degradation without a baseline.'),
'small-3456':('Calls GPU hardware instability and orders code defect established; presents memory leak or insufficient RAM as exhaustive cause.','Treats successful retries as minor observation, not a separate major incident.'),
'large-1234':('Calls GPU hardware failure confirmed; infers timeout recovery and normal operations throughout from sparse records.','No distinct invented major incident.'),
'large-2345':('Mislabels OOM evidence L901/L902 as L900/L901; says 10 seconds for 20-second SSH interval and for 30-second link-down to connected interval; asserts memory leak/resource contention confirmed.','Calls periodic successful retries latency issues without a baseline.'),
'large-3456':('Mislabels OOM evidence L901/L902 as L900/L901; says worker killed because it consumed excessive memory, treats GPU as physically detached, and infers compromised data integrity.','No distinct invented major incident.'),
'normal-1234':('Matches timeout to retry; asserts resource/health normality and underlying recovery beyond what logs establish.','No significant false incident.'),
'normal-2345':('Matches timeout to retry; claims all services healthy and timeout self-resolved without proof.','No significant false incident.'),
'normal-3456':('Matches timeout to retry; conflates inventory and API health, claims whole system self-healed in one second.','No significant false incident.'),
'ambiguous-1234':('Identifies both 504s and both retries; asserts proxy generated timeouts, outage invisible to users, normal resources and full recovery from sparse evidence.','No distinct fabricated incident, but infers full recovery and duration threshold.'),
'ambiguous-2345':('Identifies both 504s and retries; incorrectly says r100 retry followed schedule by one second rather than two; rules out resource exhaustion and asserts user operations succeeded.','Infers sustained slowness between two observations without proof.'),
'ambiguous-3456':('Identifies both 504s and retries; asserts proxy configured timeout, healthy metrics and full recovery, no user-facing errors or data loss without evidence.','No distinct fabricated incident, but infers healthy recovery without a baseline.')}
grades={}
for k,(reason,fp) in notes.items():
 x=json.loads((p/(k+'-result.json')).read_text()); assert x['finish_reason']=='stop' and x['content'].strip() and not x['reasoning'].strip()
 grades[k]={'method':'manual final-answer review against original Round 3 incident and control keys','completion':'stop','mentioned':incident if k.startswith(('small','large')) else None,'missed':[] if k.startswith(('small','large')) else None,'false_positives':fp,'evidence_and_reasoning':reason}
(p/'grades.json').write_text(json.dumps(grades,indent=2)+'\n')
def measure(dir,scenario):
 xs=[json.loads(f.read_text()) for f in sorted(dir.glob(scenario+'-*-result.json'))]
 return (statistics.median(x['timings']['prompt_per_second'] for x in xs),statistics.median(x['timings']['predicted_per_second'] for x in xs))
rows=[]
for s in ['small','large','normal','ambiguous']:
 a,b=measure(p,s),measure(baseline,s);rows.append(f'| {s} | {a[0]:.1f} / {a[1]:.1f} | {b[0]:.1f} / {b[1]:.1f} |')
report='''# MiniCPM-V-4.7-35B-A3B CPU-only — Round 4

All 12 responses finished normally with nonempty final answers and no generated reasoning. The model mentioned all six planted incident categories on each of the six incident responses. This is mention coverage, not proof of correct diagnosis: unsupported causes, impact, and evidence errors recur. No model is established as reliable for unattended monitoring by these four synthetic inputs.

## Comparison with original MiniCPM5-2B

| Measure | MiniCPM-V 4.7 | MiniCPM5-2B Round 4 |
|---|---|---|
| Completed | 12/12 | 11/12 |
| Small incident mentions, seeds 1234/2345/3456 | 6/6/6 | 6/5/6 |
| Large incident mentions, seeds 1234/2345/3456 | 6/6/6 | N/5/4 |
| Output capped at 8,192 | 0/12 | 1/12 (large 1234; ungraded) |

Both models completed all six controls. On the normal control, the new model recognized timeout and retry without inventing a major incident in all three seeds. On the ambiguous control, it identified both 504s and retries in all three, but inferred proxy behavior, stable health or full recovery from insufficient evidence; seed 2345 inferred sustained slowness. The baseline controls also overstate health, with one ambiguous seed asserting consistent 504s and isolation to orders. The comparison is not a controlled model-only experiment: the new model is non-thinking as requested, while the baseline had thinking enabled; builds, tokenizer, placement and contention also differ.

The new model repeatedly calls the Xid 79 event confirmed hardware failure or physical GPU detachment and invents recovery, root cause, or impact. Large seeds 2345 and 3456 misidentify OOM lines L901/L902 as L900/L901; large 2345 also reports incorrect intervals. Small 2345 treats successful retries as degradation without a baseline. All three ambiguous controls infer resource normality and recovery from a single process snapshot; the last says retries prevented data loss or user-facing errors without proof. Per-response manual reviews are in grades.json. A claim of a possible cause is differentiated from a definite unsupported claim; no numeric hallucination count is asserted.

## Throughput

Median over the three seeds, in prompt tokens/s / generated tokens/s:

| Scenario | MiniCPM-V 4.7 | MiniCPM5-2B baseline |
|---|---:|---:|
'''+'\n'.join(rows)+'''

These are server-reported observational rates, not matched performance benchmarks. Baseline ran concurrently with other partitions; this run used node 1 with 16 strictly pinned decode threads spread over six CCDs, 48 batch threads and process-local THP. Prompts tokenize differently: small 2800 vs 2197 tokens, large 9475 vs 7250. The capped baseline large run remains included in throughput medians and excluded from correctness scoring.

## Method and provenance

Reused the four original datasets, Round 4 task instructions, seeds 1234/2345/3456, 32,768 context, 8,192 output cap, temperature 1, top-p .95, top-k 0, min-p 0, repetition penalty 1, presence penalty 0 and cache_prompt false. Dataset SHA-256 checks passed. Prompt token counts plus output budget fit the context; server reported truncated=0 on each request. Text only; enable_thinking=false. The model chat template's empty closed thinking block was verified; no response emitted reasoning. Smoke test completed before evaluation, with stop and a sensible final answer. The original Round 4 keys and manual review rules were reused; the new grades are in grades.json.

OpenBMB publisher GGUF revision cf943fbe32f840b20ca05ccc146e7dc72baf9a9b; model SHA-256 9ecf91e814d7e9adc64325b1b65f8628a9b1bb8da4f61a92869ca7489bfd888e verified. Upstream ggml-org/llama.cpp commit 0ec4027a2a0183de87c007069adddc06e665f7de, CPU-only build; exact server command in command.json and server binary hash in server.sha256. Smoke, requests, raw streamed events, answers, rendered prompts and server log are retained here. Existing casebook service and GPU miner were left running; the temporary evaluation server was stopped after the run. No tuning trials were performed.
'''
(p/'REPORT.md').write_text(report)
