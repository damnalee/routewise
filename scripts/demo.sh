#!/usr/bin/env bash
# RouteWise end-to-end demo: quality gate, live scenarios, traffic, dashboard, screenshot.
# Run from anywhere with the routewise conda env active:  bash scripts/demo.sh
set -o pipefail
cd "$(dirname "$0")/.." || exit 1

BASE=http://localhost:8000
FAILS=0
PIDS=()

say() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
cleanup() { for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null; done; }
trap cleanup EXIT

wait_for() {
  local url="$1" secs="$2"
  for _ in $(seq 1 "$secs"); do
    curl -sf "$url" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

ask() {
  local label="$1" expect="$2" body="$3"
  echo
  echo ">>> $label   (expect: $expect)"
  if ! curl -s --max-time 120 -w '\nHTTP %{http_code}\n' "$BASE/v1/query" "${AUTH[@]}" \
      -H 'Content-Type: application/json' -d "$body" | python scripts/demo_show.py "$expect"; then
    FAILS=$((FAILS + 1))
  fi
}

python -c "import routewise" 2>/dev/null || { echo "Activate the routewise conda env first."; exit 1; }

say "1. Quality gate"
ruff check . && ruff format --check . && mypy && pytest -q 2>&1 | tail -3 || { echo "Quality gate failed."; exit 1; }

say "2. Start the gateway"
for port in 8000 8501; do
  if lsof -ti ":$port" >/dev/null 2>&1; then
    echo "Port $port is busy. Free it with:  lsof -ti :$port | xargs kill"
    exit 1
  fi
done
mkdir -p data docs
export ROUTEWISE_DB_PATH=data/demo.db
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
rm -f data/demo.db data/demo.db-wal data/demo.db-shm

python -m uvicorn routewise.app:build_app --factory --port 8000 >/tmp/routewise-gateway.log 2>&1 &
PIDS+=($!)
if ! wait_for "$BASE/healthz" 120; then
  echo "Gateway did not start. Last log lines:"
  tail -20 /tmp/routewise-gateway.log
  exit 1
fi
echo "gateway up, /healthz ok"
if ! wait_for "$BASE/readyz" 20; then
  echo "Gateway is up but /readyz failed: the LLM provider is not reachable. Check your Groq key."
  exit 1
fi
echo "provider reachable, /readyz ok"

KEY=$(python -c "from routewise.config import get_settings; print(get_settings().gateway_api_key)")
AUTH=(-H "X-API-Key: $KEY")

say "3. Live scenarios, each checked automatically"
ask "Easy question, first time" small \
  '{"query": "How can I reset my password?", "max_tokens": 1000}'
ask "Exact repeat, served from the cache" cache \
  '{"query": "How can I reset my password?", "max_tokens": 1000}'
ask "Paraphrase, hits only if similarity is at least 0.85" cache,small \
  '{"query": "What is the way to reset my password?", "max_tokens": 1000}'
ask "Celsius to Fahrenheit, first time" small \
  '{"query": "How do I convert Celsius to Fahrenheit?", "max_tokens": 1000}'
ask "Fahrenheit to Celsius: the lexical guard must block a false hit" small \
  '{"query": "How do I convert Fahrenheit to Celsius?", "max_tokens": 1000}'
ask "Same password question in another namespace: must miss" small \
  '{"query": "How can I reset my password?", "namespace": "other-kb", "max_tokens": 1000}'
ask "Hard architecture question, routed to the large model" large \
  '{"query": "Design a scalable architecture for a multi-tenant SaaS platform and compare the trade-offs between a shared database and database-per-tenant.", "max_tokens": 3000}'
ask "Same hard question again: cache hit, large cost saved" cache,large \
  '{"query": "Design a scalable architecture for a multi-tenant SaaS platform and compare the trade-offs between a shared database and database-per-tenant.", "max_tokens": 3000}'
ask "Easy question forced onto the large model" large \
  '{"query": "What is the capital of France?", "force_tier": "large", "max_tokens": 1000}'
ask "RAG: answer must use the supplied context" small \
  '{"query": "What is the refund window?", "context": ["Refunds are accepted within 30 days of purchase."], "namespace": "docs", "max_tokens": 1000}'
ask "Blank query is rejected" http422 \
  '{"query": "   "}'

if [ "$FAILS" -eq 0 ]; then
  say "All scenarios behaved as expected"
else
  say "$FAILS scenario(s) were unexpected, see [UNEXPECTED] above"
fi

say "4. Stats and Prometheus metrics"
curl -s "$BASE/v1/stats" "${AUTH[@]}" | python -m json.tool | head -40
curl -s "$BASE/metrics" \
  | grep -E '^routewise_(requests_total|cost_usd_total|savings_usd_total|cache_entries|cache_rejections|cost_pressure)'

say "5. Simulated traffic, 40 requests"
python scripts/simulate_traffic.py --n 40 --max-tokens 1000 --concurrency 2

say "6. Dashboard"
python -m streamlit run src/routewise/dashboard.py --server.headless true --server.port 8501 \
  >/tmp/routewise-dashboard.log 2>&1 &
PIDS+=($!)
if wait_for "http://localhost:8501" 60; then
  open http://localhost:8501
  echo "dashboard: http://localhost:8501  (choose 'Last hour', wait for the charts to fill)"
else
  echo "Dashboard did not start. See /tmp/routewise-dashboard.log"
  exit 1
fi

say "7. Screenshot"
echo "Manual route: press Cmd+Shift+5 now, capture the browser window, save it as docs/dashboard.png, then type s."
read -r -p "Press Enter to capture with screencapture (then press Space and click the window), or type s to skip: " choice
if [ "$choice" != "s" ]; then
  screencapture -i docs/dashboard.png
fi
if [ -f docs/dashboard.png ]; then
  echo "found docs/dashboard.png"
else
  echo "no screenshot yet"
fi
