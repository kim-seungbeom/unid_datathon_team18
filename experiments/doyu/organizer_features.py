"""운영진 v7 특징 코드와 레포 경로를 한곳에서 제공한다.

사용: `from organizer_features import M, REPO_ROOT`  (M.build_features, M.LABELS 등)
"""
from pathlib import Path

import organizer_v7_model as M  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[2]
