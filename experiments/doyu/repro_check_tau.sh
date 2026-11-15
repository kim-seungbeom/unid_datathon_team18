#!/usr/bin/env bash
# 최종 선택 CSV(submission_inner_best_40_60_prior_tau.csv) 재현 검사: test.py -> 원본 비교 -> check_submission.
# 사용법 (레포 루트에서): bash experiments/doyu/repro_check_tau.sh [원본 CSV] [데이터 폴더]
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
ORIG="${1:-/home/ubuntu/doyu/unid_prior_correction/runs/doyu/submissions/submission_inner_best_40_60_prior_tau.csv}"
[[ -f "$ORIG" ]] || ORIG=/home/ubuntu/doyu/submission_inner_best_40_60_prior_tau.csv
D="${2:-/home/ubuntu/data/battery_data_light}"
OUT="outputs/repro_tau_$(date +%Y%m%d_%H%M%S).csv"
[[ -f "$ORIG" ]] || { echo "원본 CSV가 없습니다: $ORIG" >&2; exit 1; }

echo "원본 CSV: $ORIG"
echo "== 1) test.py (규정 형식)"
python test.py --input "$D/test/test.csv" --image_root "$D/test" --output "$OUT"
echo "== 2) 원본 CSV와 비교"
python experiments/doyu/tau_final.py compare "$ORIG" "$OUT"
echo "== 3) 재현성 검사 (test.py 여러 번 실행, 오래 걸림)"
python check_submission.py --data_dir "$D" --final "$ORIG"
echo "== 완료: 모든 단계 통과"
