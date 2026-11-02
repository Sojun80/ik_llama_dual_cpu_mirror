#!/bin/bash
# perfrec.sh OUT_DIR BIN ENVSTR CASE [perf-record args...]
# Starts one server (baseline layout, 32 decode threads), sends the CASE corpus request,
# records perf on the server during decode, then stops everything.
set -u
OUT=$1; BIN=$2; ENVSTR=$3; CASE=$4; shift 4
R=/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round5
KI=/home/ubuntu/llm-evaluations/20261031-minicpm-v47-kernel-investigation
MODEL=/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4/model.complete.gguf
ORDER=48,49,52,56,57,60,64,65,68,72,73,76,80,84,88,92,81,89,50,58,66,74,82,90,51,59,67,75,83,91,53,61,69,77,85,93,54,62,70,78,86,94,55,63,71,79,87,95
PORT=19186
mkdir -p "$OUT"
env GLIBC_TUNABLES=glibc.malloc.hugetlb=1 $ENVSTR numactl --physcpubind=48-95 --membind=1 "$BIN" -m "$MODEL" -ngl 0 --no-mmap \
    -t 32 -tb 48 --cpu-range $ORDER -c 16384 --host 127.0.0.1 --port $PORT --parallel 1 > "$OUT/server.log" 2>&1 &
SRV=$!
for i in $(seq 600); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; sleep 1; done
python3 - "$KI/corpus-$CASE.txt" $PORT > "$OUT/request.json" <<'PY' &
import json, sys, urllib.request
txt = open(sys.argv[1]).read()
req = urllib.request.Request(f'http://127.0.0.1:{sys.argv[2]}/completion',
      json.dumps({'prompt': txt, 'temperature': 0, 'n_predict': 512, 'cache_prompt': False, 'ignore_eos': True}).encode(),
      {'Content-Type': 'application/json'})
print(json.dumps(json.load(urllib.request.urlopen(req, timeout=900))['timings']))
PY
REQ=$!
# wait until decode has started: prompt processing takes ~7 s (medium) / ~27 s (long)
if [ "$CASE" = long ]; then sleep 32; else sleep 11; fi
sudo -n perf record -o "$OUT/perf.data" -p $SRV "$@" -- sleep 6 > "$OUT/perf-record.log" 2>&1
wait $REQ
kill $SRV; wait $SRV 2>/dev/null
sudo -n chown ubuntu:ubuntu "$OUT/perf.data"
cat "$OUT/request.json"
