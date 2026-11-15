#!/usr/bin/env bash
# Run manually on the organizer's server. Installs only into a personal environment.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENSEMBLE_BASE_PYTHON="${ENSEMBLE_BASE_PYTHON:-/home/ubuntu/venv/bin/python}"
ENSEMBLE_VENV="${ENSEMBLE_VENV:-$HOME/seungbeom/venv-ensemble}"

if [[ ! -x "$ENSEMBLE_BASE_PYTHON" ]]; then
    echo "공용 Python을 찾지 못했습니다. ENSEMBLE_BASE_PYTHON에 기존 서버 Python을 지정하세요." >&2
    exit 1
fi

# A venv created from another venv does not inherit its site-packages automatically.
# Append the shared packages explicitly, behind the personal venv's own packages.
"$ENSEMBLE_BASE_PYTHON" - "$ENSEMBLE_VENV" <<'PY'
import json
import pathlib
import subprocess
import sys
import sysconfig

if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10 이상이 필요합니다.")
shared_prefix = pathlib.Path(sys.prefix).resolve()
shared_packages = pathlib.Path(sysconfig.get_path("purelib")).resolve()
target = pathlib.Path(sys.argv[1]).expanduser().resolve()
if target == shared_prefix or shared_prefix in target.parents or target in shared_prefix.parents:
    raise SystemExit("개인 환경은 공용 환경과 별개의 폴더여야 합니다.")
if target.exists() and not (target / "pyvenv.cfg").is_file():
    raise SystemExit("대상 폴더가 이미 있고 가상환경이 아닙니다. ENSEMBLE_VENV에 새 경로를 지정하세요.")
if not (target / "bin" / "python").is_file():
    subprocess.run([sys.executable, "-m", "venv", str(target)], check=True)
python = target / "bin" / "python"
info = json.loads(subprocess.check_output([
    str(python), "-c",
    "import json,sys,sysconfig; print(json.dumps({'prefix':sys.prefix,'site':sysconfig.get_path('purelib'),'version':list(sys.version_info[:2])}))",
], text=True))
if pathlib.Path(info["prefix"]).resolve() != target:
    raise SystemExit("개인 Python의 환경 경로가 예상과 다릅니다. 설치를 중단합니다.")
if tuple(info["version"]) != sys.version_info[:2]:
    raise SystemExit("개인 환경과 공용 환경의 Python 주·부 버전이 다릅니다. 새 ENSEMBLE_VENV 경로를 지정하세요.")
personal_packages = pathlib.Path(info["site"]).resolve()
if target not in personal_packages.parents:
    raise SystemExit("개인 패키지 설치 경로가 환경 폴더 밖입니다. 설치를 중단합니다.")
(personal_packages / "99_organizer_shared_packages.pth").write_text(
    str(shared_packages) + "\n", encoding="utf-8"
)
print("개인 환경:", target)
print("공용 패키지 참조:", shared_packages)
PY

"$ENSEMBLE_VENV/bin/python" -m pip --isolated --require-virtualenv install --no-user \
    -r "$SCRIPT_DIR/requirements.txt"

# Confirm that personal package installation did not replace the shared CUDA pair.
"$ENSEMBLE_VENV/bin/python" - "$ENSEMBLE_BASE_PYTHON" <<'PY'
import importlib.util
import json
import pathlib
import subprocess
import sys

packages = ("torch", "torchvision")
expected = json.loads(subprocess.check_output([
    sys.argv[1], "-c",
    "import importlib.util,json; print(json.dumps({n:(s.origin if (s:=importlib.util.find_spec(n)) else None) for n in ('torch','torchvision')}))",
], text=True))
for name in packages:
    spec = importlib.util.find_spec(name)
    if spec is None or expected[name] is None:
        raise SystemExit(f"공용 환경의 {name} 설치를 확인하세요. 개인 환경에서 임의로 교체하지 않았습니다.")
    if pathlib.Path(spec.origin).resolve() != pathlib.Path(expected[name]).resolve():
        raise SystemExit(f"개인 환경에 별도 {name}가 있습니다. 새 ENSEMBLE_VENV 경로로 다시 실행하세요.")

import joblib
import lightgbm
import sklearn
import torch
import torchvision
import xgboost

print("패키지 확인 완료:", "sklearn", sklearn.__version__, "xgboost", xgboost.__version__,
      "lightgbm", lightgbm.__version__, "joblib", joblib.__version__)
print("공용 CUDA 패키지:", "torch", torch.__version__, "torchvision", torchvision.__version__)
print("CUDA 사용 가능:", torch.cuda.is_available())
PY

echo
echo "완료. 서버 터미널에서 활성화하세요:"
printf 'source %q/bin/activate\n' "$ENSEMBLE_VENV"
