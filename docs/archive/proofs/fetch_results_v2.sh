#!/bin/bash
# Poll Aristotle and download completed results (batch 3: theorems 16-22)
export ARISTOTLE_API_KEY="arstl_ZVHoKYGC0gq15cVSA0YRRaQEvZ9mT5vUE3dpOei3Dgs"

# Job ID → theorem name mapping
declare -a JOB_IDS=(
  "6ab6d9bf-63e5-40cd-9b32-dd59fa806af2"
  "afd79399-273e-4b4e-a66b-e71383022243"
  "71b55cc7-dc26-48ac-893a-628fe72a8e75"
  "d94b3ae2-ab4d-4a26-b7fc-fc6e5975e019"
  "4833c5a3-4603-41c5-8b90-de36ba8b124b"
  "6e1e254b-cfa1-4601-96be-39791ba48d0a"
  "849e1849-4dd4-453f-a1a2-c7d680f35d67"
)
declare -a NAMES=(
  "theorem16-optimal-top-n-trades"
  "theorem17-gate-sortino-with-semivariance"
  "theorem18-frequency-optimum-under-5bps"
  "theorem19-approximation-gain-vs-estimation-cost"
  "theorem20-min-hold-barrier-win-rate-surface"
  "theorem21-loss-cluster-counterexample"
  "theorem22-dual-gate-dependence-bound"
)

echo "=== Aristotle Batch 3 Status ==="
aristotle list
echo ""

for i in "${!JOB_IDS[@]}"; do
  id="${JOB_IDS[$i]}"
  name="${NAMES[$i]}"
  status=$(aristotle list 2>&1 | grep "$id" | awk '{print $2}')
  if [[ "$status" == "COMPLETE"* ]]; then
    echo "✓ $name ($id) — $status"
    if [ ! -f "./proofs/${name}.tar.gz" ]; then
      aristotle result "$id" --destination "./proofs/${name}.tar.gz"
      echo "  Saved to ./proofs/${name}.tar.gz"
    else
      echo "  Already downloaded"
    fi
  else
    echo "⏳ $name ($id) — ${status:-UNKNOWN}"
  fi
done
