"""Generate the 25 synthetic Euler maps, label them with the crystal-plasticity proxy and split them 20 / 5.

Presets:
  --preset paper : 800 x 1000 px at 0.5 um (the paper's EBSD maps), 100 px patches (resized to 128 for training)
  --preset quick : 512 x 640 px at 0.78125 um (same 400 x 500 um field), 64 px patches -> CPU-friendly
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from msdesign import plotting as P  # noqa: E402
from msdesign.cp_proxy import properties  # noqa: E402
from msdesign.synthetic import CONDITIONS, make_dataset  # noqa: E402

PRESETS = {'paper': dict(H=800, W=1000, pixel_um=0.5, patch=100),
           'quick': dict(H=512, W=640, pixel_um=0.78125, patch=64)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', choices=list(PRESETS), default='quick')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    cfg = PRESETS[args.preset]
    out = args.out or os.path.join(os.path.dirname(__file__), '..', 'work', args.preset)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    images, states = make_dataset(seed=args.seed, H=cfg['H'], W=cfg['W'], pixel_um=cfg['pixel_um'])
    print(f'rendered {len(images)} maps in {time.time() - t0:.0f}s')
    real = np.genfromtxt(os.path.join(os.path.dirname(__file__), '..', 'data', 'official', 'dataset.csv'),
                         delimiter=',', skip_header=1)[:, 4:6]
    props = []
    for i, img in enumerate(images):
        ys, el = properties(img, pixel_um=cfg['pixel_um'])
        props.append((ys, el))
        print(f'{i + 1:2d} {CONDITIONS[i]}  X_rx={states[i]["x_rx"]:.2f}  proxy YS={ys:6.1f} EL={el:5.1f}   '
              f'(measured {real[i, 0]:.0f} / {real[i, 1]:.1f})', flush=True)
    props = np.array(props)
    rng = np.random.default_rng(args.seed)
    test_idx = np.sort(rng.choice(25, 5, replace=False))
    train_idx = np.setdiff1d(np.arange(25), test_idx)
    np.savez_compressed(os.path.join(out, 'dataset.npz'), images=(images * 255).round().astype(np.uint8),
                        props=props, conditions=np.array(CONDITIONS), train_idx=train_idx, test_idx=test_idx,
                        pixel_um=cfg['pixel_um'], patch=cfg['patch'])
    info = dict(preset=args.preset, **cfg, train_idx=train_idx.tolist(), test_idx=test_idx.tolist(), states=states,
                corr_with_measured=dict(ys=float(np.corrcoef(props[:, 0], real[:, 0])[0, 1]),
                                        el=float(np.corrcoef(props[:, 1], real[:, 1])[0, 1])))
    with open(os.path.join(out, 'dataset_info.json'), 'w') as f:
        json.dump(info, f, indent=1)
    print('correlation with measured properties:', info['corr_with_measured'])

    P.setup()
    plt = P.plt
    fig, axs = plt.subplots(5, 5, figsize=(12, 10))
    for i, ax in enumerate(axs.ravel()):
        ax.imshow(images[i], interpolation='nearest')
        T, t, r = CONDITIONS[i]
        ax.set_title(f'#{i + 1} {r}%/{T}C/{t}h  YS {props[i, 0]:.0f}  EL {props[i, 1]:.0f}', fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'synthetic_maps.png'), dpi=110)
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.scatter(real[:, 0], real[:, 1], s=30, c=P.BLUE, edgecolors='white', label='measured (paper data)')
    ax.scatter(props[:, 0], props[:, 1], s=30, c=P.ORANGE, marker='s', edgecolors='white',
               label='synthetic maps + CP proxy')
    ax.set_xlabel('Yield strength (MPa)')
    ax.set_ylabel('Elongation (%)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'synthetic_vs_measured_properties.png'))


if __name__ == '__main__':
    main()
