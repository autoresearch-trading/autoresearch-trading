#!/bin/bash
# Poll Aristotle and download completed results
export ARISTOTLE_API_KEY="arstl_ZVHoKYGC0gq15cVSA0YRRaQEvZ9mT5vUE3dpOei3Dgs"

declare -A JOBS=(
  ["theorem0-math-review"]="c2104bdd-1f58-4f63-8e58-558f5e8f84b4"
  ["theorem1-sufficient-statistic"]="0409f430-9563-4515-b597-d9298e94964f"
  ["theorem2-entropy-collapse"]="60d40941-c1a4-4dad-b4d0-193b607fb519"
  ["theorem3-kelly-barriers"]="919eb95d-4db8-4e6c-b1d9-ba3395cf98ff"
  ["theorem4-hawkes-predictability"]="141128e3-cef1-435c-86fe-668dd15174e6"
  ["theorem5-architecture-bounds"]="73ee445b-2c68-4d8a-938c-f0e28cabfdaf"
)

echo "=== Aristotle Job Status ==="
aristotle list
echo ""

for name in "${!JOBS[@]}"; do
  id="${JOBS[$name]}"
  status=$(aristotle list 2>&1 | grep "$id" | awk '{print $2}')
  if [ "$status" = "COMPLETED" ]; then
    echo "✓ $name ($id) — COMPLETED, downloading..."
    aristotle result "$id" --destination "./proofs/${name}.tar.gz"
    echo "  Saved to ./proofs/${name}.tar.gz"
  else
    echo "⏳ $name ($id) — $status"
  fi
done
