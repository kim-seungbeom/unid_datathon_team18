"""Shared CPU HGB predictor and frozen image inference; no observation logging."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path, PurePosixPath

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_sample_weight

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import features as F
import postprocess as P
from evaluate import CONFIGS, make_model

SEED = 42


class SubmissionError(ValueError):
    """Safe error: never include observations, identifiers or image paths."""


def config_from_name(name='base', post_mode='mask_cummax', seed=SEED, balanced=False):
    if name not in CONFIGS or post_mode not in P.MODES:
        raise SubmissionError('Invalid model configuration.')
    cfg = CONFIGS[name]
    return dict(config_name=name, feature_cols=F.feature_columns(cfg['events'], cfg.get('gas', False), cfg.get('cond', False)),
                drop=list(cfg.get('drop', [])), img_pcs=cfg.get('img', 0),
                img_delta=cfg.get('delta', False), post_mode=post_mode,
                seed=seed, balanced=balanced, hgb_params={})


def encoder_module():
    """Prefer the original image_features module; fall back to the identical adjacent copy
    (the submission repository may ship final_model/ alone)."""
    for folder in (HERE,):
        source = folder / 'encoders.py'
        if source.is_file():
            key = '_unid_submission_image_encoders'
            if key not in sys.modules:
                sys.path.insert(0, str(folder))
                spec = importlib.util.spec_from_file_location(key, source)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.modules[key] = module
            return sys.modules[key]
    raise SubmissionError('The image encoder dependency is unavailable.')


def validate_frame(frame):
    required = {'sample_id', 'experiment_id', 'relative_time_s', *F.SENSORS}
    if not required.issubset(frame.columns) or len(frame) == 0:
        raise SubmissionError('Input is empty or required columns are missing.')
    if frame.columns.duplicated().any():
        raise SubmissionError('Duplicate input columns.')
    for col in ('sample_id', 'experiment_id'):
        ids = frame[col].astype('string')
        if ids.isna().any() or ids.str.strip().eq('').any():
            raise SubmissionError('Missing identifiers.')
    if frame['sample_id'].astype(str).duplicated().any():
        raise SubmissionError('Duplicate sample identifiers.')


def resolve_images(frame, image_root):
    try:
        if 'image_path' not in frame or frame['image_path'].isna().any():
            raise SubmissionError('Missing image references.')
        root = Path(image_root).resolve()
        out = []
        for value in frame['image_path'].astype(str):
            rel = PurePosixPath(value.replace('\\', '/'))
            if rel.is_absolute() or '..' in rel.parts or ':' in value or not rel.name:
                raise SubmissionError('Invalid relative image reference.')
            candidates = (root.joinpath(*rel.parts), root / rel.name, root / 'images' / rel.name)
            hit = None
            for candidate in candidates:
                resolved = candidate.resolve()
                if not resolved.is_relative_to(root):
                    raise SubmissionError('Invalid relative image reference.')
                if resolved.is_file():
                    hit = resolved
                    break
            if hit is None:
                raise SubmissionError('An input image is unavailable.')
            out.append(hit)
        return out
    except Exception:
        raise SubmissionError('Unable to resolve input images.') from None


def embed_images(frame, image_root, device='cpu', batch_size=32, *, _encoder=None):
    """ResNet18 IMAGENET1K_V1, 224 letterbox, output in input row order.

    _encoder is an offline-test injection point, unavailable in the submission CLI.
    """
    try:
        import torch
        from PIL import Image
        if device not in ('cpu', 'cuda') or batch_size < 1:
            raise SubmissionError('Invalid image inference settings.')
        torch.manual_seed(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        encoders = encoder_module()
        enc = (_encoder if _encoder is not None else encoders.build_encoder('resnet18', pretrained=True))
        enc = enc.to(device).eval()
        paths = resolve_images(frame, image_root)
        out = np.empty((len(paths), 512), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(paths), batch_size):
                batch = []
                for path in paths[start:start + batch_size]:
                    with Image.open(path) as im:
                        batch.append(encoders.preprocess(im))
                result = enc(torch.stack(batch).to(device)).float().cpu().numpy()
                if result.shape != (len(batch), 512) or not np.isfinite(result).all():
                    raise SubmissionError('Invalid image embedding.')
                out[start:start + len(batch)] = result
        return out
    except Exception:
        raise SubmissionError('Image embedding failed; check inputs and encoder installation locally.') from None


class Predictor:
    def __init__(self, config=None):
        self.config = config_from_name() if config is None else dict(config)
        self.config.setdefault('seed', SEED)
        self.config.setdefault('drop', [])
        self.config.setdefault('img_pcs', 0)
        self.config.setdefault('img_delta', False)
        self.config.setdefault('post_mode', 'mask_cummax')
        self.config.setdefault('hgb_params', {})
        if self.config['post_mode'] not in P.MODES or self.config['img_pcs'] < 0:
            raise SubmissionError('Invalid predictor configuration.')
        self.pca = None
        self.model = None

    def _embeddings(self, embeddings, n):
        try:
            e = np.asarray(embeddings, dtype=np.float32)
            if e.shape != (n, 512) or not np.isfinite(e).all():
                raise SubmissionError('Invalid embeddings.')
            return e
        except Exception:
            raise SubmissionError('Expected finite 512-dimensional image embeddings.') from None

    def _matrix(self, frame, embeddings):
        validate_frame(frame)
        feats = F.build_features(frame)
        cols = [c for c in self.config['feature_cols'] if c not in self.config['drop']]
        if not cols or not set(cols).issubset(feats.columns):
            raise SubmissionError('Invalid feature selection.')
        X = feats[cols].reset_index(drop=True).copy()
        k = self.config['img_pcs']
        if k:
            if self.pca is None:
                raise SubmissionError('The image projection is not fitted.')
            Z = self.pca.transform(self._embeddings(embeddings, len(frame)))
            X[[f'img_pc{i}' for i in range(k)]] = Z
            if self.config['img_delta']:
                X['img_change30s'] = F.image_change(frame, Z)
        return X, feats

    def fit(self, frame, labels, embeddings=None):
        """Fit only on the caller's training partition; never used during inference."""
        validate_frame(frame)
        y = np.asarray(labels)
        if y.shape != (len(frame),) or not np.isin(y, np.arange(1, 7)).all():
            raise SubmissionError('Training labels must be integer stages from 1 to 6.')
        k = self.config['img_pcs']
        if k:
            e = self._embeddings(embeddings, len(frame))
            if not isinstance(k, int) or k > min(e.shape):
                raise SubmissionError('Invalid image PCA dimension.')
            self.pca = PCA(n_components=k, random_state=self.config['seed']).fit(e)
        X, _ = self._matrix(frame, embeddings)
        self.model = make_model(self.config['seed'], self.config['hgb_params'])
        weight = compute_sample_weight('balanced', y) if self.config.get('balanced') else None
        self.model.fit(X, y.astype(int), sample_weight=weight)
        return self

    def predict_proba(self, frame, embeddings=None):
        if self.model is None:
            raise SubmissionError('The predictor is not fitted.')
        X, feats = self._matrix(frame, embeddings)
        probabilities = np.zeros((len(frame), 6))
        classes = np.asarray(self.model.classes_, dtype=int)
        if len(classes) == 1:
            # HGB returns a binary-shaped probability array even for one class.
            probabilities[:, classes[0] - 1] = 1.0
        else:
            probabilities[:, classes - 1] = self.model.predict_proba(X)
        return probabilities, feats

    def predict(self, frame, embeddings=None):
        proba, feats = self.predict_proba(frame, embeddings)
        return P.apply(proba, feats, frame['experiment_id'].astype(str).to_numpy(),
                       frame['relative_time_s'].to_numpy(float), self.config['post_mode']).astype(int)

    def save(self, directory):
        if self.model is None:
            raise SubmissionError('The predictor is not fitted.')
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        meta = dict(self.config)
        sources = [HERE / f for f in ('model.py', 'train.py', 'test.py', 'features.py', 'postprocess.py', 'evaluate.py')]
        if self.config['img_pcs']:
            sources.append(Path(encoder_module().__file__))
        meta['code_sha256'] = {('image_features/encoders.py' if p.name == 'encoders.py' else p.name):
                               hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        meta['package_versions'] = {}
        for name in ('numpy', 'pandas', 'scikit-learn', 'joblib', 'torch', 'torchvision', 'Pillow'):
            try:
                meta['package_versions'][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                meta['package_versions'][name] = 'not-installed'
        meta['python_version'] = platform.python_version()
        meta['platform'] = platform.system()
        meta['hgb_resolved_params'] = self.model.get_params()
        meta['encoder'] = {'name': 'resnet18', 'weights': 'IMAGENET1K_V1',
                           'preprocess': 'image_features.encoders.preprocess; letterbox224',
                           'software_license': 'BSD-3-Clause'} if self.config['img_pcs'] else None
        joblib.dump({'pca': self.pca, 'model': self.model}, directory / 'model.joblib')
        (directory / 'config.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, directory):
        """Load trusted, team-created artifacts only (joblib uses pickle).
        A blend directory (blend.json) loads as BlendPredictor."""
        if (Path(directory) / 'blend.json').is_file():
            return BlendPredictor.load(directory)
        directory = Path(directory)
        cfg = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
        predictor = cls(cfg)
        state = joblib.load(directory / 'model.joblib')
        predictor.pca, predictor.model = state['pca'], state['model']
        if predictor.model is None or (cfg['img_pcs'] and predictor.pca is None):
            raise SubmissionError('Incomplete model artifacts.')
        return predictor


def parse_blend(spec):
    """'base:0.5,base_img16_delta:0.5' -> [('base', 0.5), ...] (weights normalized)."""
    try:
        items = [(n.strip(), float(w)) for n, w in (p.split(':') for p in spec.split(','))]
    except Exception:
        raise SubmissionError('Blend spec must look like name:weight,name:weight.') from None
    if not items or any(n not in CONFIGS or w <= 0 for n, w in items) or len({n for n, _ in items}) != len(items):
        raise SubmissionError('Invalid blend members or weights.')
    total = sum(w for _, w in items)
    return [(n, w / total) for n, w in items]


class BlendPredictor:
    """Weighted average of member probabilities, then one causal post-processing step.

    Selected on OOF (Train+Val, 5-fold GroupKFold, 3 seeds): base + base_img16_delta 50:50, mask_cummax.
    """

    def __init__(self, members, post_mode='mask_cummax', seed=SEED, balanced=False):
        if post_mode not in P.MODES:
            raise SubmissionError('Invalid post-processing mode.')
        self.weights = [w for _, w in members]
        self.members = [Predictor(config_from_name(n, 'none', seed, balanced)) for n, _ in members]
        self.post_mode = post_mode
        self.config = {'blend': [(n, w) for n, w in members], 'post_mode': post_mode, 'seed': seed,
                       'img_pcs': max(m.config['img_pcs'] for m in self.members)}
        if not self.config['img_pcs']:
            raise SubmissionError('The final model must use image inputs.')

    def fit(self, frame, labels, embeddings=None):
        for m in self.members:
            m.fit(frame, labels, embeddings if m.config['img_pcs'] else None)
        return self

    def predict_proba(self, frame, embeddings=None):
        total, feats = None, None
        for w, m in zip(self.weights, self.members):
            p, feats = m.predict_proba(frame, embeddings if m.config['img_pcs'] else None)
            total = w * p if total is None else total + w * p
        return total, feats

    def predict(self, frame, embeddings=None):
        proba, feats = self.predict_proba(frame, embeddings)
        return P.apply(proba, feats, frame['experiment_id'].astype(str).to_numpy(),
                       frame['relative_time_s'].to_numpy(float), self.post_mode).astype(int)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        names = []
        for (name, _), m in zip(self.config['blend'], self.members):
            m.save(directory / f'member_{name}')
            names.append(name)
        (directory / 'blend.json').write_text(json.dumps(self.config, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        cfg = json.loads((directory / 'blend.json').read_text(encoding='utf-8'))
        obj = cls.__new__(cls)
        obj.config = cfg
        obj.post_mode = cfg['post_mode']
        obj.weights = [w for _, w in cfg['blend']]
        obj.members = [Predictor.load(directory / f'member_{n}') for n, _ in cfg['blend']]
        if not cfg['img_pcs'] or P.MODES.count(obj.post_mode) != 1:
            raise SubmissionError('Incomplete blend artifacts.')
        return obj
