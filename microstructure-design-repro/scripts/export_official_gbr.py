"""Convert the authors' pickled GradientBoosting models (scikit-learn 0.23) into a portable .npz tree dump.

The pickles in model_save/ of the official repository can only be unpickled with scikit-learn <= 1.2 and cannot be
used for prediction by any recent version. Run this once in an environment with scikit-learn 1.2.x, e.g.

    python -m venv venv_sk12 && venv_sk12/bin/pip install scikit-learn==1.2.2 "numpy<2" joblib
    venv_sk12/bin/python scripts/export_official_gbr.py --model-dir <official repo>/model_save

The resulting data/official/gbr_official_trees.npz is read by msdesign.inverse.TreeEnsemble.
"""
import argparse
import os

import joblib
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', 'data', 'official',
                                                  'gbr_official_trees.npz'))
    args = ap.parse_args()
    arrays = {}
    for name in ('GBR_ys', 'GBR_el'):
        search = joblib.load(os.path.join(args.model_dir, name + '.pkl'))
        est = search.best_estimator_
        trees = [t.tree_ for t in est.estimators_.ravel()]
        arrays[f'{name}_init'] = np.array(est.init_.constant_.ravel()[0])
        arrays[f'{name}_lr'] = np.array(est.learning_rate)
        arrays[f'{name}_n_nodes'] = np.array([t.node_count for t in trees])
        for key in ('children_left', 'children_right', 'feature', 'threshold'):
            arrays[f'{name}_{key}'] = np.concatenate([getattr(t, key) for t in trees])
        arrays[f'{name}_value'] = np.concatenate([t.value.ravel() for t in trees])
        arrays[f'{name}_train_n'] = np.array(trees[0].n_node_samples[0])
        arrays[f'{name}_best_params'] = np.array(str(search.best_params_))
        arrays[f'{name}_cv_best_score'] = np.array(search.best_score_)
    np.savez_compressed(args.out, **arrays)
    print('saved', args.out)


if __name__ == '__main__':
    main()
