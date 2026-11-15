"""Team consistency checker, not the organizer's official script. No row logging."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


def read_output(path, ids):
    out = pd.read_csv(path, dtype={'sample_id': str})
    if list(out.columns) != ['sample_id', 'label']:
        raise ValueError('Invalid output columns.')
    if out.sample_id.tolist() != ids or not out.sample_id.is_unique:
        raise ValueError('Invalid output identifiers or ordering.')
    if not pd.api.types.is_integer_dtype(out.label) or not out.label.between(1, 6).all():
        raise ValueError('Invalid output labels.')
    return out.set_index('sample_id').label


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--final', type=Path, required=True)
    args = ap.parse_args()
    candidates = [args.data_dir / 'test' / 'test.csv', args.data_dir / 'test.csv']
    source = next((p for p in candidates if p.is_file()), None)
    if source is None:
        raise ValueError('Test input is unavailable.')
    frame = pd.read_csv(source, dtype={'sample_id': str, 'experiment_id': str})
    baseline = None
    with tempfile.TemporaryDirectory(prefix='unid-check-') as folder:
        tmp = Path(folder)

        def run(part):
            src, dst = tmp / 'input.csv', tmp / 'output.csv'
            part.to_csv(src, index=False)
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('test.py')),
                '--input', str(src), '--image_root', str(source.parent), '--output', str(dst)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode:
                raise ValueError('Inference execution failed.')
            out = read_output(dst, part.sample_id.tolist())
            if baseline is not None and not np.array_equal(out, baseline.loc[out.index]):
                raise ValueError('Predictions changed for shared inputs.')
            return out

        baseline = run(frame)
        final = read_output(args.final, frame.sample_id.tolist())
        if not np.array_equal(baseline, final):
            raise ValueError('Final file differs from test.py output.')
        print('PASS final reproduction')
        run(frame)
        print('PASS repeated input')
        run(frame.sample(frac=1, random_state=42))
        print('PASS shuffled rows')
        for _, group in frame.groupby('experiment_id', sort=False):
            run(group)
        print('PASS experiment isolation')
        for _, group in frame.groupby('experiment_id', sort=False):
            group = group.sort_values('relative_time_s', kind='stable')
            run(group.iloc[:max(1, int(len(group) * .6))])
        print('PASS future removal')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        raise SystemExit('FAIL self-check; inspect locally without sharing data.') from None
