"""Final fitting on Train + Validation; stdout contains row counts and time only."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import model as M
from evaluate import CONFIGS
from postprocess import MODES


def find_csv(data_dir, split):
    for candidate in (data_dir / split / f'{split}.csv', data_dir / f'{split}.csv'):
        if candidate.is_file():
            return candidate
    raise M.SubmissionError('A training split is unavailable.')


def load_embeddings(emb_dir, split, ids):
    """Align a complete, private extraction cache by sample_id, never positional guesswork."""
    try:
        folder = Path(emb_dir) / split
        meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        if meta.get('status') != 'complete':
            raise M.SubmissionError('Incomplete embeddings.')
        if meta.get('model', meta.get('encoder', 'resnet18')) != 'resnet18':
            raise M.SubmissionError('Incompatible image encoder.')
        if 'weights' in meta and meta['weights'] != 'torchvision:ResNet18_Weights.IMAGENET1K_V1':
            raise M.SubmissionError('Incompatible encoder weights.')
        if meta.get('l2_normalize', False):
            raise M.SubmissionError('Normalized embeddings are incompatible with image inference.')
        order = pd.read_csv(folder / 'sample_ids.csv', dtype={'sample_id': str})['sample_id']
        if order.isna().any() or order.str.strip().eq('').any() or not order.is_unique:
            raise M.SubmissionError('Invalid embedding identifiers.')
        values = np.load(folder / 'features.npy', mmap_mode='r', allow_pickle=False)
        if values.shape != (len(order), 512):
            raise M.SubmissionError('Invalid embedding dimensions.')
        position = pd.Index(order).get_indexer(ids.astype(str))
        if (position < 0).any():
            raise M.SubmissionError('Missing embeddings.')
        result = np.asarray(values[position], dtype=np.float32)
        if not np.isfinite(result).all():
            raise M.SubmissionError('Invalid embeddings.')
        return result
    except Exception:
        raise M.SubmissionError('Unable to load complete compatible image embeddings.') from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--config', choices=CONFIGS, default=None,
                        help='단일 모델 학습. 생략하면 --blend(최종 선택)를 학습')
    parser.add_argument('--post_mode', choices=MODES, default='mask_cummax')
    parser.add_argument('--emb_dir', type=Path)
    parser.add_argument('--out', type=Path, default=Path(M.__file__).resolve().parent / 'weights')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--balanced', action='store_true')
    parser.add_argument('--blend', default='base:0.5,base_img16_delta:0.5',
                        help="최종 선택 블렌드 (--config를 주면 무시)")
    parser.add_argument('--splits', default='train,validation',
                        help="학습에 쓸 split. 'train'만 주면 공식 Validation 재현용")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    blend = None if args.config else M.parse_blend(args.blend)
    if blend:
        predictor = M.BlendPredictor(blend, args.post_mode, args.seed, args.balanced)
        cfg = predictor.config
    else:
        cfg = M.config_from_name(args.config, args.post_mode, args.seed, args.balanced)
        predictor = M.Predictor(cfg)
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    if not splits or not set(splits) <= {'train', 'validation'}:
        raise M.SubmissionError('Invalid split selection.')
    frames, embeddings = [], []
    for split in splits:
        source = find_csv(args.data_dir, split)
        frame = pd.read_csv(source, dtype={'sample_id': str, 'experiment_id': str})
        M.validate_frame(frame)
        if 'label' not in frame:
            raise M.SubmissionError('Training labels are missing.')
        if cfg['img_pcs']:
            emb = (load_embeddings(args.emb_dir, split, frame['sample_id']) if args.emb_dir else
                   M.embed_images(frame, source.parent, args.device, args.batch_size))
            embeddings.append(emb)
        frames.append(frame)
    # Official validation is distinct by experiment: prevent accidental split mixing.
    if len(frames) == 2 and set(frames[0].experiment_id) & set(frames[1].experiment_id):
        raise M.SubmissionError('Training and validation experiments overlap.')
    frame = pd.concat(frames, ignore_index=True)
    emb = np.concatenate(embeddings) if embeddings else None
    predictor.fit(frame, frame['label'].to_numpy(), emb)
    predictor.save(args.out)
    print(f'rows={len(frame)} seconds={time.perf_counter() - started:.3f}')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        raise SystemExit('Training failed; inspect inputs and installation locally.') from None
