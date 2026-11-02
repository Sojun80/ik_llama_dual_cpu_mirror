#!/usr/bin/env python3
"""Round-5 speculative-decoding test: realistic chat prompts instead of r5bench's random-word corpus (which no draft
can predict). Starts one server per config, runs every prompt at temperature 0, records decode speed, draft stats and
the output text so speculative and plain runs can be compared token for token.

usage: spectest.py --out runs/NN-name --rounds R --config name=<extra server args separated by spaces, or "none"> ...
All configs use the mirror m64 layout and the recommended IK_R5 environment.
"""
import argparse, json, os, pathlib, random, signal, subprocess, sys, time, urllib.request

R = pathlib.Path('/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round5')
MODEL = R.parent / '20261031-minicpm-v47-round4/model.complete.gguf'
BIN = R / 'build-mirror3/bin/llama-server'
PORT = 19186
ENV = dict(IK_R5_DYN='1', IK_R5_FA='1', IK_R5_HUGE='1', IK_R5_PINONCE='1', IK_R5_FMOE='1', IK_R5_FMOE_UG='1',
           IK_R5_NONUMAB='1')


def synth_logs(n=40):
    rnd = random.Random(7)
    units = ['sshd', 'kernel', 'systemd', 'nginx', 'smartd', 'cron', 'docker', 'nvidia-persistenced']
    msgs = ['Accepted publickey for ubuntu from 192.168.1.%d port %d ssh2', 'Started Daily apt download activities.',
            'nvme0n1: I/O error, dev nvme0n1, sector %d op 0x0:(READ)', 'upstream timed out (110: Connection timed out) while reading response header',
            'Device: /dev/sda [SAT], 8 Currently unreadable (pending) sectors', 'pam_unix(cron:session): session opened for user root',
            'container %d exited with code 137 (OOM killed)', 'NVRM: Xid 79, GPU has fallen off the bus.',
            'Failed password for invalid user admin from 45.12.%d.%d port %d ssh2', 'EXT4-fs warning: mounting fs with errors, running e2fsck is recommended']
    out = []
    for i in range(n):
        m = rnd.choice(msgs)
        m = m.replace('%d', '{}').format(*[rnd.randint(1, 60000) for _ in range(m.count('%d'))])
        out.append(f'Nov  1 {8 + i // 6:02d}:{(i * 7) % 60:02d}:{(i * 13) % 60:02d} gigabyte5 {rnd.choice(units)}[{rnd.randint(300, 9000)}]: {m}')
    return '\n'.join(out)


PROMPTS = {
    'code': 'Write a Python function that parses one line of an nginx access log (combined format) into a dict with '
            'typed fields, plus pytest unit tests covering normal lines, IPv6 clients and malformed input.',
    'logs': 'Here are syslog lines from a server:\n\n' + synth_logs() + '\n\nList every problem you see as a JSON array '
            'of objects with fields "time", "unit", "severity" (info/warning/error/critical) and "summary". Output only the JSON.',
    'prose': 'Explain in about 350 words how CPU caches work, for a programmer who knows what RAM is but has never '
             'thought about cache lines, associativity or prefetching.',
}


def http(path, obj=None, timeout=900):
    data = json.dumps(obj).encode() if obj is not None else None
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}' + path, data, {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def run(cfg_name, extra, out, rnd, n_predict):
    stock = extra[:1] == ['stock']   # stock code path: node 1 only, no mirror, no IK_R5 fusions
    if stock:
        extra = extra[1:]
        cmd = ['numactl', '--physcpubind=48-95', '--membind=1', str(BIN), '-m', str(MODEL), '-ngl', '0', '--no-mmap',
               '-t', '48', '-tb', '48', '-c', '16384', '--host', '127.0.0.1', '--port', str(PORT), '--parallel', '1'] + extra
        env = dict(os.environ)
    else:
        cpus = (R / 'tools/m64.txt').read_text().strip()
        cmd = ['numactl', '--physcpubind=16-95', str(BIN), '-m', str(MODEL), '-ngl', '0', '--no-mmap', '--numa', 'mirror',
               '--numa-mirror', 'all', '-t', '64', '-tb', '64', '--cpu-range', cpus, '-c', '16384', '--host', '127.0.0.1',
               '--port', str(PORT), '--parallel', '1'] + extra
        env = dict(os.environ, **ENV)
    tag = f'{cfg_name}-r{rnd}'
    recs = []
    with open(out / f'server-{tag}.log', 'w') as log:
        srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            for _ in range(600):
                if srv.poll() is not None:
                    raise RuntimeError(f'server exited {srv.returncode}')
                try:
                    urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2); break
                except Exception:
                    time.sleep(1)
            http('/v1/chat/completions', {'messages': [{'role': 'user', 'content': 'Say hello.'}], 'max_tokens': 16, 'temperature': 0})
            for name, text in PROMPTS.items():
                x = http('/v1/chat/completions', {'messages': [{'role': 'user', 'content': text}], 'max_tokens': n_predict,
                                                  'temperature': 0, 'seed': 1234, 'cache_prompt': False})
                t = x.get('timings', {})
                rec = dict(config=cfg_name, round=rnd, prompt=name, timings=t, content=x['choices'][0]['message'].get('content'),
                           reasoning=x['choices'][0]['message'].get('reasoning_content'), finish=x['choices'][0].get('finish_reason'))
                recs.append(rec)
                acc = f" draft {t.get('draft_n_accepted', 0)}/{t.get('draft_n', 0)}" if t.get('draft_n') else ''
                print(f"{tag:14s} {name:6s} decode={t.get('predicted_per_second', 0):7.3f} n={t.get('predicted_n')}{acc}", flush=True)
        finally:
            try:
                os.killpg(srv.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                srv.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(srv.pid, signal.SIGKILL); srv.wait()
    (out / f'run-{tag}.json').write_text(json.dumps(recs, indent=1))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--config', action='append', required=True)
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--n-predict', type=int, default=512)
    a = ap.parse_args()
    out = R / a.out
    out.mkdir(parents=True, exist_ok=False)
    cfgs = []
    for c in a.config:
        n, v = c.split('=', 1)
        cfgs.append((n, [] if v == 'none' else v.split()))
    (out / 'args.json').write_text(json.dumps(dict(vars(a), prompts=PROMPTS), indent=1))
    allr = []
    for rnd in range(a.rounds):
        order = cfgs if rnd % 2 == 0 else cfgs[::-1]
        for n, extra in order:
            allr += run(n, extra, out, rnd, a.n_predict)
    print('SUMMARY')
    base = cfgs[0][0]
    for n, _ in cfgs:
        for p in PROMPTS:
            rs = [r for r in allr if r['config'] == n and r['prompt'] == p]
            d = [r['timings'].get('predicted_per_second', 0) for r in rs]
            dn = sum(r['timings'].get('draft_n', 0) for r in rs); da = sum(r['timings'].get('draft_n_accepted', 0) for r in rs)
            same = all(r['content'] == next(b['content'] for b in allr if b['config'] == base and b['prompt'] == p and b['round'] == r['round'])
                       and r['reasoning'] == next(b['reasoning'] for b in allr if b['config'] == base and b['prompt'] == p and b['round'] == r['round'])
                       for r in rs)
            print(f"{n:10s} {p:6s} decode mean {sum(d) / len(d):7.3f} {[round(x, 2) for x in d]}"
                  + (f"  accepted {da}/{dn} = {da / dn:.0%}" if dn else '') + f"  same_as_{base}={same}")
    (out / 'summary.json').write_text(json.dumps(allr, indent=1))
    print('DONE')


if __name__ == '__main__':
    main()
