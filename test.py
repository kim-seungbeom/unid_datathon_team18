"""Inference from the supplied CSV and its images only; never fits a model."""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import model as M


def submission_frame(frame, labels):
    if 'sample_id' not in frame:
        raise M.SubmissionError('Sample identifiers are missing.')
    ids = frame['sample_id'].astype('string')
    if ids.isna().any() or ids.str.strip().eq('').any() or not ids.is_unique:
        raise M.SubmissionError('Sample identifiers must be present and unique.')
    labels = np.asarray(labels)
    if labels.shape != (len(frame),) or not np.isin(labels, np.arange(1, 7)).all():
        raise M.SubmissionError('Prediction count or label range is invalid.')
    return pd.DataFrame({'sample_id': frame['sample_id'].to_numpy(), 'label': labels.astype(np.int64)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--image_root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, default=Path(M.__file__).resolve().parent / 'weights')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    if args.input.resolve() == args.output.resolve():
        raise M.SubmissionError('Output must differ from input.')
    frame = pd.read_csv(args.input, dtype={'sample_id': str, 'experiment_id': str})
    M.validate_frame(frame)
    predictor = M.Predictor.load(args.artifacts)
    emb = (M.embed_images(frame, args.image_root, args.device, args.batch_size)
           if predictor.config['img_pcs'] else None)
    out = submission_frame(frame, predictor.predict(frame, emb))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f'rows={len(out)} seconds={time.perf_counter() - started:.3f}')
    return out


if __name__ == '__main__':
    try:
        main()
    except Exception:
        raise SystemExit('Inference failed; inspect inputs, artifacts and installation locally.') from None
