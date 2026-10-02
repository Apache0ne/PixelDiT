"""Compute FID and Inception Score for generated PNG images."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256_of(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def score_images(images, fid_stats, expected_count=50000):
    images, fid_stats = Path(images), Path(fid_stats)
    if not images.is_dir():
        raise ValueError(f'Image directory does not exist: {images}')
    if not fid_stats.is_file():
        raise ValueError(f'FID statistics file does not exist: {fid_stats}')
    paths = sorted(images.glob('*.png'))
    if len(paths) != expected_count:
        raise ValueError(f'Expected {expected_count} PNG images, found {len(paths)}')
    unexpected = [p.name for p in images.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.png'} and p.suffix != '.png']
    if unexpected:
        raise ValueError('Image directory must contain only the generated PNG image set')

    import torch_fidelity

    metrics = torch_fidelity.calculate_metrics(
        input1=str(images),
        input2=None,
        fid_statistics_file=str(fid_stats),
        cuda=True,
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=False,
    )
    fid = float(metrics['frechet_inception_distance'])
    inception_score = float(metrics['inception_score_mean'])
    if not math.isfinite(fid) or not math.isfinite(inception_score):
        raise ValueError('Metric calculation produced a non-finite result')
    return {
        'status': 'ok',
        'image_count': len(paths),
        'image_dir': str(images.resolve()),
        'fid_statistics_file': str(fid_stats.resolve()),
        'fid_statistics_sha256': sha256_of(fid_stats),
        'fid': fid,
        'inception_score': inception_score,
        'torch_fidelity_version': getattr(torch_fidelity, '__version__', 'unknown'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--fid-stats', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fid-stats-sha256', help='Expected SHA256 of the FID statistics file')
    parser.add_argument('--expected-count', type=int, default=50000)
    args = parser.parse_args()
    if args.fid_stats_sha256 and (not args.fid_stats.is_file() or sha256_of(args.fid_stats) != args.fid_stats_sha256):
        parser.error('FID statistics SHA256 does not match')
    result = score_images(args.images, args.fid_stats, args.expected_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
