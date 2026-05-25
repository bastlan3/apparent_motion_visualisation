import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from motion_render import MotionDataset


def save_frames_as_png(frames, out_dir='demo_frames'):
    try:
        import numpy as np
        from PIL import Image
        os.makedirs(out_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            arr = (frame.numpy() * 255).clip(0, 255).astype('uint8')
            Image.fromarray(arr, mode='L').save(f'{out_dir}/frame_{i:03d}.png')
        print(f'Saved {len(frames)} frames to {out_dir}/')
    except ImportError:
        print('Pillow not installed — skipping PNG save (pip install Pillow)')


def main():
    print('Creating MotionDataset...')
    dataset = MotionDataset(n_sequences=100, T=16, image_height=64, image_width=64, seed=42)
    print(f'  Dataset size : {len(dataset)} sequences')

    seq = dataset[0]
    print(f'  Sequence shape : {tuple(seq.shape)}  (T x H x W)')
    print(f'  Value range    : [{seq.min():.3f}, {seq.max():.3f}]')
    print(f'  Non-zero pixels in frame 0: {(seq[0] > 0).sum().item()}')

    save_frames_as_png(seq, 'demo_frames')

    print('\nTesting DataLoader...')
    loader = DataLoader(dataset, batch_size=4, num_workers=0, shuffle=True)
    batch = next(iter(loader))
    print(f'  Batch shape: {tuple(batch.shape)}  (B x T x H x W)')

    print('\nTesting reproducibility...')
    seq_a = dataset[0]
    seq_b = dataset[0]
    print(f'  Identical on repeat call: {torch.allclose(seq_a, seq_b)}')

    print('\nAll checks passed.')


if __name__ == '__main__':
    main()
