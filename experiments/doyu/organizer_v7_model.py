"""baselines/organizer_v7/model.py(운영진 v7 특징 코드)를 수정 없이 이 모듈 이름으로 실행한다.

레포 루트 model.py(팀 FusionClassifier)와 이름이 겹치지 않도록 별도 모듈명을 쓴다.
일반 import 가능한 모듈이어야 joblib 병렬 worker가 _image_stats 등을 다시 찾을 수 있다.
"""
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[2] / 'baselines' / 'organizer_v7' / 'model.py'
exec(compile(_SOURCE.read_text(encoding='utf-8'), str(_SOURCE), 'exec'))
