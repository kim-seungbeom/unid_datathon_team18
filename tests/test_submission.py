"""Synthetic-only submission contract tests."""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model as M
import postprocess as P
import train as T
from test_features import synth

spec = importlib.util.spec_from_file_location('submission_cli', Path(M.__file__).with_name('test.py'))
CLI = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CLI)


@pytest.fixture(scope='module')
def bundle():
    with tempfile.TemporaryDirectory(prefix='unid-submission-') as folder:
        root = Path(folder)
        frame = pd.concat([synth(f'synthetic-{i}', n=60, seed=i) for i in range(4)], ignore_index=True)
        labels = np.tile(np.repeat(np.arange(1, 7), 10), 4)
        emb = np.random.default_rng(42).normal(size=(len(frame), 512)).astype('float32')
        emb[:, 0] += labels * 3
        yield root, frame, labels, emb


@pytest.mark.parametrize('mode', P.MODES)
def test_prediction_contract(bundle, mode):
    root, frame, labels, emb = bundle
    predictor = M.Predictor(M.config_from_name('events_img16_delta', post_mode=mode)).fit(frame, labels, emb)
    before = predictor.pca.components_.copy()
    baseline = pd.Series(predictor.predict(frame, emb), index=frame.sample_id)
    np.testing.assert_array_equal(baseline, predictor.predict(frame, emb))
    order = np.random.default_rng(7).permutation(len(frame))
    shuffled = pd.Series(predictor.predict(frame.iloc[order], emb[order]), index=frame.sample_id.iloc[order])
    pd.testing.assert_series_equal(baseline, shuffled.reindex(baseline.index))
    for idx in frame.groupby('experiment_id').indices.values():
        for part in (idx, idx[:len(idx) // 2]):
            np.testing.assert_array_equal(predictor.predict(frame.iloc[part], emb[part]), baseline.iloc[part])
    np.testing.assert_array_equal(before, predictor.pca.components_)
    artifact = root / mode
    predictor.save(artifact)
    np.testing.assert_array_equal(baseline, M.Predictor.load(artifact).predict(frame, emb))
    meta = json.loads((artifact / 'config.json').read_text())
    assert meta['seed'] == 42 and meta['package_versions']['scikit-learn']
    assert all(len(value) == 64 for value in meta['code_sha256'].values())


def test_base_has_no_pca_and_missing_classes_are_supported(bundle):
    _, frame, _, _ = bundle
    cfg = M.config_from_name('base', post_mode='none')
    pred = M.Predictor(cfg).fit(frame, np.full(len(frame), 3), None)
    assert pred.pca is None and cfg['img_pcs'] == 0
    np.testing.assert_array_equal(pred.predict(frame, None), np.full(len(frame), 3))


def test_image_embedding_and_csv_pipeline(bundle, monkeypatch):
    import torch
    from PIL import Image
    torch.set_num_threads(1)
    encoders = M.encoder_module()
    torch.manual_seed(42)
    encoder = encoders.build_encoder('resnet18', pretrained=False)
    root, frame, labels, emb = bundle
    image_root = root / 'png'
    image_root.mkdir()
    part = frame.iloc[np.r_[np.arange(0, 60, 5), np.arange(60, 120, 5)]].copy()
    part['sample_id'] = [f'00{i}' for i in range(len(part))]
    part['image_path'] = [f'nested/{i}.png' for i in range(len(part))]
    for i in range(len(part)):
        Image.new('RGB', (30 + i, 20), ((i * 30) % 256, 50, 100)).save(image_root / f'{i}.png')
    actual_embed = M.embed_images
    def offline(fr, image_root, device='cpu', batch_size=32):
        return actual_embed(fr, image_root, device, batch_size, _encoder=encoder)
    a = offline(part, image_root, batch_size=4)
    b = offline(part.iloc[::-1], image_root, batch_size=1)
    assert a.shape == (len(part), 512) and np.isfinite(a).all()
    np.testing.assert_allclose(a, b[::-1], rtol=1e-4, atol=1e-5)
    M.Predictor(M.config_from_name('events_img16_delta')).fit(frame, labels, emb).save(root / 'cli-artifacts')
    monkeypatch.setattr(M, 'embed_images', offline)
    src, dst = root / 'input.csv', root / 'submission.csv'
    part.to_csv(src, index=False)
    CLI.main(['--input', str(src), '--image_root', str(image_root), '--output', str(dst),
              '--artifacts', str(root / 'cli-artifacts'), '--device', 'cpu'])
    out = pd.read_csv(dst, dtype={'sample_id': str})
    assert out.columns.tolist() == ['sample_id', 'label']
    assert out.sample_id.tolist() == part.sample_id.tolist()
    assert out.label.between(1, 6).all() and pd.api.types.is_integer_dtype(out.label)
    baseline = out.set_index('sample_id')['label']
    pieces = [part, part.sample(frac=1, random_state=8)]
    for _, group in part.groupby('experiment_id', sort=False):
        pieces.extend([group, group.iloc[:len(group) // 2]])
    for subset in pieces:
        subset.to_csv(src, index=False)
        CLI.main(['--input', str(src), '--image_root', str(image_root), '--output', str(dst),
                  '--artifacts', str(root / 'cli-artifacts'), '--device', 'cpu'])
        repeated = pd.read_csv(dst, dtype={'sample_id': str}).set_index('sample_id')['label']
        pd.testing.assert_series_equal(repeated, baseline.loc[repeated.index])
    bad = part.copy()
    bad.loc[bad.index[0], 'image_path'] = 'private-missing.png'
    with pytest.raises(M.SubmissionError) as error:
        offline(bad, image_root)
    assert 'private-missing' not in str(error.value)


@pytest.mark.parametrize('kind', ['duplicate', 'missing', 'fractional', 'range', 'length'])
def test_submission_rejects_invalid_output(bundle, kind):
    _, frame, _, _ = bundle
    frame = frame.iloc[:3].copy()
    labels = np.array([1., 2., 3.])
    if kind == 'duplicate':
        frame.loc[frame.index[1], 'sample_id'] = frame.sample_id.iloc[0]
    elif kind == 'missing':
        frame.loc[frame.index[1], 'sample_id'] = None
    elif kind == 'fractional':
        labels[0] = 1.5
    elif kind == 'range':
        labels[0] = 7
    else:
        labels = labels[:2]
    with pytest.raises(M.SubmissionError):
        CLI.submission_frame(frame, labels)


def test_train_cli_nested_layout_and_embedding_alignment(bundle, capsys):
    root, frame, labels, emb = bundle
    data, cache, artifacts = root / 'synthetic-data', root / 'synthetic-emb', root / 'trained'
    for split, positions in (('train', np.arange(180)), ('validation', np.arange(180, 240))):
        fr = frame.iloc[positions].copy()
        fr['label'] = labels[positions]
        (data / split).mkdir(parents=True)
        fr.to_csv(data / split / f'{split}.csv', index=False)
        d = cache / split
        d.mkdir(parents=True)
        np.save(d / 'features.npy', emb[positions][::-1])
        pd.DataFrame({'sample_id': fr.sample_id.iloc[::-1]}).to_csv(d / 'sample_ids.csv', index=False)
        (d / 'metadata.json').write_text(json.dumps({'status': 'complete', 'encoder': 'resnet18'}))
    np.testing.assert_array_equal(T.load_embeddings(cache, 'train', frame.sample_id.iloc[:180]), emb[:180])
    T.main(['--data_dir', str(data), '--emb_dir', str(cache), '--config', 'events_img16_delta',
            '--post_mode', 'none', '--out', str(artifacts)])
    assert M.Predictor.load(artifacts).config['post_mode'] == 'none'
    assert 'synthetic-' not in capsys.readouterr().out


def test_normalized_cache_is_rejected(bundle):
    root, frame, _, emb = bundle
    cache = root / 'normalized-cache'
    folder = cache / 'train'
    folder.mkdir(parents=True)
    np.save(folder / 'features.npy', emb)
    frame[['sample_id']].to_csv(folder / 'sample_ids.csv', index=False)
    (folder / 'metadata.json').write_text(json.dumps({'status': 'complete', 'l2_normalize': True}))
    with pytest.raises(M.SubmissionError):
        T.load_embeddings(cache, 'train', frame.sample_id)


def test_encoder_fallback_to_adjacent_copy(monkeypatch, tmp_path):
    """image_features 폴더가 없어도 final_model/encoders.py로 인코더를 찾는다."""
    import sys as _sys
    from pathlib import Path as _P
    import model as _M
    fake = tmp_path / "final_model"
    fake.mkdir()
    monkeypatch.setattr(_M, "HERE", fake)
    (fake / "encoders.py").write_bytes((_P(_M.__file__).parent / "encoders.py").read_bytes())
    _sys.modules.pop("_unid_submission_image_encoders", None)
    mod = _M.encoder_module()
    assert _P(mod.__file__).resolve() == (fake / "encoders.py").resolve()
    _sys.modules.pop("_unid_submission_image_encoders", None)
