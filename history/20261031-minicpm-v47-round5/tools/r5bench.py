#!/usr/bin/env python3
"""Round-5 benchmark harness.

Runs one or more server configs against the medium (2834 tok) and long (9455 tok)
corpora with the baseline payload (temperature 0, n_predict 512, ignore_eos,
cache_prompt false). One server process per (config, case, round) so configs can be
interleaved to control drift; only one model instance is ever alive.

  r5bench.py --out DIR --config NAME=BIN:THREADS:CPULIST[:ENV=V,...] [...] \
             [--cases medium,long] [--rounds 2] [--probs] [--prof] [--perf] [--n-predict 512]

CPULIST is the full ordered --cpu-range (decode uses its first THREADS entries,
batch uses 48). Use the literal 'base' for the baseline order.
"""
import argparse, json, os, pathlib, signal, subprocess, sys, time, urllib.request

ROOT  = pathlib.Path('/home/ubuntu/llm-evaluations')
KI    = ROOT / '20261031-minicpm-v47-kernel-investigation'
MODEL = ROOT / '20261031-minicpm-v47-round4/model.complete.gguf'
BASE_ORDER = '48,49,52,56,57,60,64,65,68,72,73,76,80,84,88,92,81,89,50,58,66,74,82,90,51,59,67,75,83,91,53,61,69,77,85,93,54,62,70,78,86,94,55,63,71,79,87,95'
EXPECT = {'medium': 2834, 'long': 9455}
PORT = 19185
FORBIDDEN = set(range(0, 16))   # casebook llama-server cores; never touch
# R5_CASEBOOK_OFF=1 (user-approved 2026-11-01: casebook may be stopped for benchmarks): cores 0-15 allowed, but only
# while casebook-llm.service is not running
if os.environ.get("R5_CASEBOOK_OFF") == "1":
    _st = subprocess.run(["systemctl", "is-active", "casebook-llm.service"], capture_output=True, text=True).stdout.strip()
    assert _st != "active", "R5_CASEBOOK_OFF=1 but casebook-llm.service is active"
    FORBIDDEN = set()

PERF_EVENTS = ['cycles', 'instructions', 'task-clock', 'context-switches', 'cpu-migrations',
               'ls_refills_from_sys.ls_mabresp_lcl_dram', 'ls_refills_from_sys.ls_mabresp_lcl_cache',
               'l2_cache_req_stat.ic_dc_miss_in_l2', 'ls_l1_d_tlb_miss.all', 'l2_dtlb_misses' ]
if os.environ.get('R5_PERF_EVENTS'):   # override, comma separated
    PERF_EVENTS = os.environ['R5_PERF_EVENTS'].split(',')


def parse_config(s):
    name, rest = s.split('=', 1)
    parts = rest.split(':')
    binary, threads, cpus = parts[0], int(parts[1]), parts[2]
    env = {}
    if len(parts) > 3 and parts[3]:
        for kv in parts[3].split(','):
            k, v = kv.split('=', 1)
            env[k] = v
    if cpus == 'base':
        cpus = BASE_ORDER
    cl = [int(x) for x in cpus.split(',')]
    mode = env.pop('R5_MODE', 'node1')
    mirror = env.pop('R5_MIRROR', 'weights')
    argsfile = env.pop('R5_ARGSFILE', '')
    extra = [l.strip() for l in pathlib.Path(argsfile).read_text().splitlines() if l.strip()] if argsfile else []
    assert len(set(cl)) == len(cl) and not (set(cl) & FORBIDDEN), 'cpus must be distinct and avoid 0-15'
    if mode == 'mirror':
        # block split: first half of the list must be node-0 cpus (16-47), second half node 1 (48-95)
        assert len(cl) == threads and threads % 2 == 0
        assert all((16 if FORBIDDEN else 0) <= c <= 47 for c in cl[:threads//2]) and all(48 <= c <= 95 for c in cl[threads//2:])
    else:
        assert len(cl) == 48 and all(48 <= c <= 95 for c in cl)
    return dict(name=name, bin=binary, threads=threads, cpus=cpus, env=env, mode=mode, mirror=mirror, extra=extra)


def http(base, path, obj=None, timeout=900):
    data = json.dumps(obj).encode() if obj is not None else None
    req = urllib.request.Request(base + path, data, {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def stream_completion(base, payload, on_token):
    """Streaming /completion; calls on_token(i) per generated chunk; returns final chunk."""
    payload = dict(payload, stream=True)
    req = urllib.request.Request(base + '/completion', json.dumps(payload).encode(), {'Content-Type': 'application/json'})
    last, i, text = None, 0, []
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b'data:'):
                continue
            obj = json.loads(line[5:])
            text.append(obj.get('content', ''))
            if not obj.get('stop'):
                on_token(i); i += 1
            last = obj
    last['content_full'] = ''.join(text)
    return last


def run_one(cfg, case, rnd, args, out):
    tag = f"{cfg['name']}-{case}-r{rnd}"
    env = os.environ.copy()
    env['GLIBC_TUNABLES'] = 'glibc.malloc.hugetlb=1'
    env.update(cfg['env'])
    if args.prof:
        env['IK_R5_PROF'] = str(out / f'prof-{tag}.txt')
    if cfg['mode'] == 'mirror':
        cmd = ['numactl', '--physcpubind=' + ('16-95' if FORBIDDEN else '0-95'), cfg['bin'], '-m', str(MODEL), '-ngl', '0', '--no-mmap',
               '--numa', 'mirror', '--numa-mirror', cfg['mirror'],
               '-t', str(cfg['threads']), '-tb', str(cfg['threads']), '--cpu-range', cfg['cpus'], '-c', '16384',
               '--host', '127.0.0.1', '--port', str(PORT), '--parallel', '1']
    else:
        cmd = ['numactl', '--physcpubind=48-95', '--membind=1', cfg['bin'], '-m', str(MODEL), '-ngl', '0', '--no-mmap',
               '-t', str(cfg['threads']), '-tb', '48', '--cpu-range', cfg['cpus'], '-c', '16384',
               '--host', '127.0.0.1', '--port', str(PORT), '--parallel', '1']
    cmd += cfg.get('extra', [])
    base = f'http://127.0.0.1:{PORT}'
    txt = (KI / f'corpus-{case}.txt').read_text()
    rec = dict(tag=tag, config=cfg, case=case, round=rnd, cmd=cmd, start=time.time())
    with open(out / f'server-{tag}.log', 'w') as log:
        srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            for _ in range(600):
                if srv.poll() is not None:
                    raise RuntimeError(f'server exited {srv.returncode}')
                try:
                    urllib.request.urlopen(base + '/health', timeout=2); break
                except Exception:
                    time.sleep(1)
            else:
                raise RuntimeError('startup timeout')
            try:
                rec['numastat'] = subprocess.run(['numastat', '-p', str(srv.pid)], capture_output=True, text=True).stdout[-400:]
            except Exception:
                pass
            toks = http(base, '/tokenize', {'content': txt, 'add_special': False})['tokens']
            assert len(toks) == EXPECT[case], (case, len(toks))
            # warmup: short prompt, short decode
            http(base, '/completion', {'prompt': 'Warm up the engine with a short sentence.', 'temperature': 0,
                                       'n_predict': 32, 'cache_prompt': False, 'ignore_eos': True})
            payload = {'prompt': txt, 'temperature': 0, 'n_predict': args.n_predict, 'seed': 1234,
                       'cache_prompt': False, 'ignore_eos': True}
            if args.probs:
                payload['n_probs'] = 5
            if args.perf:
                perf = {'p': None}
                pf = out / f'perf-{tag}.txt'
                def on_token(i):
                    if i == 16:
                        perf['t0'] = time.time()
                        if os.environ.get('R5_PERF_RECORD'):   # fault addresses during decode + the address map
                            (out / f'maps-{tag}.txt').write_text(open(f'/proc/{srv.pid}/maps').read())
                            perf['p'] = subprocess.Popen(['sudo', '-n', 'perf', 'record', '-q', '-d', '-e', os.environ['R5_PERF_RECORD'],
                                                          '-o', str(out / f'perf-{tag}.data'), '-p', str(srv.pid)],
                                                         start_new_session=True)
                        else:
                            perf['p'] = subprocess.Popen(['sudo', '-n', 'perf', 'stat', '-x', ',', '-o', str(pf),
                                                          '-e', ','.join(PERF_EVENTS), '-p', str(srv.pid)],
                                                         start_new_session=True)
                    elif i == args.n_predict - 16 and perf['p']:
                        perf['p'].send_signal(signal.SIGINT); perf['t1'] = time.time(); perf['n'] = i - 16
                x = stream_completion(base, payload, on_token)
                if perf['p']:
                    perf['p'].wait(timeout=30)
                    rec['perf'] = dict(file=pf.name, seconds=perf['t1'] - perf['t0'], tokens=perf['n'])
                x['content'] = x.pop('content_full')
            else:
                x = http(base, '/completion', payload)
            rec['timings'] = x.get('timings')
            rec['content'] = x.get('content')
            if args.probs:
                rec['completion_probabilities'] = x.get('completion_probabilities')
            rec['ok'] = True
        except Exception as e:
            rec['ok'] = False; rec['error'] = repr(e)
        finally:
            try:
                os.killpg(srv.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                srv.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(srv.pid, signal.SIGKILL); srv.wait()
    (out / f'run-{tag}.json').write_text(json.dumps(rec, indent=1))
    t = rec.get('timings') or {}
    print(f"{tag:40s} ok={rec['ok']} decode={t.get('predicted_per_second', 0):.3f} prompt={t.get('prompt_per_second', 0):.2f}"
          + (f" err={rec.get('error')}" if not rec['ok'] else ''), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--config', action='append', required=True)
    ap.add_argument('--cases', default='medium,long')
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--n-predict', type=int, default=512)
    ap.add_argument('--probs', action='store_true')
    ap.add_argument('--prof', action='store_true')
    ap.add_argument('--perf', action='store_true')
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfgs = [parse_config(c) for c in args.config]
    (out / 'args.json').write_text(json.dumps(vars(args), indent=1))
    results = []
    for rnd in range(args.rounds):
        order = cfgs if rnd % 2 == 0 else cfgs[::-1]   # ABBA interleave
        for case in args.cases.split(','):
            for cfg in order:
                results.append(run_one(cfg, case, rnd, args, out))
    summary = {}
    for r in results:
        if r['ok']:
            summary.setdefault(f"{r['config']['name']}|{r['case']}", []).append(
                (round(r['timings']['predicted_per_second'], 3), round(r['timings']['prompt_per_second'], 2)))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print('SUMMARY')
    for k, v in summary.items():
        d = [a for a, _ in v]; p = [b for _, b in v]
        print(f'{k:40s} decode mean {sum(d)/len(d):.3f} {d}  prompt mean {sum(p)/len(p):.2f}')
    (out / 'DONE').write_text('done\n')


if __name__ == '__main__':
    main()
