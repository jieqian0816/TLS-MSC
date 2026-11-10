"""Paper-aligned training, validation-only selection, and held-out testing.

Default: IP, ten seeds, paper budgets, gamma=1.0 (a configurable starting value).
Use --gamma-values to select ONE gamma per dataset from mean validation accuracy
across runs. No test score is computed until this selection is complete.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import scipy
from scipy.io import loadmat
import sklearn
from sklearn.preprocessing import StandardScaler
import skimage
import torch
from torch.nn import functional as F

import model
import slic
from visual import (get_Samples_GT, assert_disjoint_splits, split_counts,
                    classification_metrics, Draw_Classification_Map, plot_test_tsne)


PAPER_CONFIG = {
    'IP': dict(folder='Indian_pines', image='Indian_pines_corrected.mat',
               image_key='indian_pines_corrected', labels='Indian_pines_gt.mat',
               label_key='indian_pines_gt', classes=16, dim=50, scale=60, epochs=200),
    'PU': dict(folder='PaviaU', image='PaviaU.mat', image_key='paviaU',
               labels='PaviaU_gt.mat', label_key='paviaU_gt', classes=9,
               dim=50, scale=300, epochs=100),
    'SA': dict(folder='Salinas', image='Salinas_corrected.mat', image_key='salinas_corrected',
               labels='Salinas_gt.mat', label_key='salinas_gt', classes=16,
               dim=50, scale=300, epochs=100),
    'HU': dict(folder='Houston', image='houston_corrected.mat', image_key='houston',
               labels='houston_gt.mat', label_key=None, classes=15,
               dim=40, scale=600, epochs=150),
}
DEFAULT_SEEDS = list(range(41, 51))


def write_json(path, obj):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                                    default=convert, allow_nan=False), encoding='utf-8')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def load_scene(data_root, dataset):
    config = PAPER_CONFIG[dataset]
    folder = Path(data_root) / config['folder']
    image_path, label_path = folder / config['image'], folder / config['labels']
    if not image_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(f'{dataset}: required data files are {image_path} and {label_path}')
    image = np.asarray(loadmat(image_path)[config['image_key']], dtype=np.float32)
    label_data = loadmat(label_path)
    if dataset == 'HU':
        # The manuscript deliberately uses a random within-scene repartition of
        # the union, rather than the official Houston challenge train/test split.
        if 'houston_gt' in label_data:
            gt = label_data['houston_gt']
        else:
            tr, te = label_data['houston_gt_tr'], label_data['houston_gt_te']
            if np.any((tr > 0) & (te > 0) & (tr != te)):
                raise ValueError('Houston source masks contain conflicting overlapping labels')
            gt = np.where(tr > 0, tr, te)
    else:
        gt = label_data[config['label_key']]
    gt = np.asarray(gt, dtype=np.int64)
    if image.ndim != 3 or gt.shape != image.shape[:2] or not np.isfinite(image).all():
        raise ValueError('Image/ground-truth shapes or spectral values are invalid')
    if not np.array_equal(np.unique(gt[gt > 0]), np.arange(1, config['classes'] + 1)):
        raise ValueError('Ground truth does not contain the expected contiguous classes')
    return image, gt


def is_better_checkpoint(val_correct, val_loss, best_correct, best_loss):
    """Accuracy is primary; loss breaks exact accuracy ties. Test data absent."""
    return val_correct > best_correct or (val_correct == best_correct and val_loss < best_loss)


def select_gamma(validation_results):
    """One fixed gamma per dataset, selected exclusively from validation logs."""
    grouped = {}
    for record in validation_results:
        grouped.setdefault(record['gamma'], []).append(record)
    return min(grouped, key=lambda gamma: (
        -np.mean([r['best_val_oa'] for r in grouped[gamma]]),
        np.mean([r['best_val_loss'] for r in grouped[gamma]]), gamma))


def create_model(config, Q, segments, train_gt, top_k, device):
    return model.GCN_in_CNN(*train_gt.shape, config['bands'], config['dim'],
                            config['classes'], torch.as_tensor(Q, device=device),
                            torch.as_tensor(segments, dtype=torch.long, device=device),
                            torch.as_tensor(train_gt, dtype=torch.long, device=device),
                            top_k=top_k).to(device)


def train_candidate(image_tensor, adjacency, Q, segments, train_gt, val_gt,
                    config, seed, gamma, top_k, checkpoint_dir, device):
    """Training and checkpoint selection deliberately have no test_gt argument."""
    set_seed(seed)  # Identical initialization for gamma candidates of a seed.
    net = create_model(config, Q, segments, train_gt, top_k, device)
    train_idx = torch.as_tensor(np.flatnonzero(train_gt.ravel()), device=device)
    val_idx = torch.as_tensor(np.flatnonzero(val_gt.ravel()), device=device)
    train_targets = torch.as_tensor(train_gt.ravel()[train_idx.cpu().numpy()] - 1,
                                     dtype=torch.long, device=device)
    val_targets = torch.as_tensor(val_gt.ravel()[val_idx.cpu().numpy()] - 1,
                                   dtype=torch.long, device=device)
    optimizer = torch.optim.Adam((p for p in net.parameters() if p.requires_grad), lr=5e-4)
    best_correct, best_loss, best_epoch = -1, float('inf'), 0
    best_state = None
    history = []
    synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, config['epochs'] + 1):
        net.train()
        optimizer.zero_grad(set_to_none=True)
        logits, *_ = net.forward_logits(adjacency, image_tensor)
        # Eqs. (24)--(25): summed CE, only labeled TRAINING pixels.
        train_loss = F.cross_entropy(logits[train_idx], train_targets, reduction='sum')
        if not torch.isfinite(train_loss):
            raise FloatingPointError(f'Nonfinite training loss at epoch {epoch}')
        train_loss.backward()
        optimizer.step()
        net.eval()
        with torch.no_grad():
            logits, *_ = net.forward_logits(adjacency, image_tensor)
            validation_loss = F.cross_entropy(logits[val_idx], val_targets, reduction='sum').item()
            validation_correct = int((logits[val_idx].argmax(1) == val_targets).sum().item())
            training_oa = float((logits[train_idx].argmax(1) == train_targets).float().mean().item())
        if not np.isfinite(validation_loss):
            raise FloatingPointError(f'Nonfinite validation loss at epoch {epoch}')
        validation_oa = validation_correct / len(val_idx)
        history.append(dict(epoch=epoch, train_loss=train_loss.item(), train_oa=training_oa,
                            val_loss=validation_loss, val_oa=validation_oa))
        if is_better_checkpoint(validation_correct, validation_loss, best_correct, best_loss):
            best_correct, best_loss, best_epoch = validation_correct, validation_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == config['epochs']:
            print(f'seed={seed} gamma={gamma:g} epoch={epoch}/{config["epochs"]} '
                  f'train_OA={training_oa * 100:.2f}% val_OA={validation_oa * 100:.2f}%', flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - start
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (checkpoint_dir / 'history.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    torch.save(best_state, checkpoint_dir / 'best_state.pt')
    result = dict(seed=seed, gamma=gamma, best_epoch=best_epoch,
                  best_val_oa=best_correct / len(val_idx), best_val_loss=best_loss,
                  training_seconds=elapsed, checkpoint=str(checkpoint_dir / 'best_state.pt'),
                  parameters=sum(p.numel() for p in net.parameters()),
                  trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad))
    write_json(checkpoint_dir / 'validation.json', result)
    del net, optimizer, best_state, logits, train_loss
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def load_saved_state(path):
    # weights_only is supported by PyTorch >=1.13; files contain only tensors.
    return torch.load(path, map_location='cpu', weights_only=True)


def evaluate_selected(image_tensor, adjacency, Q, segments, train_gt, test_gt,
                      config, selected, top_k, run_dir, device, plots):
    """Called only AFTER checkpoint AND dataset-level gamma have been selected."""
    net = create_model(config, Q, segments, train_gt, top_k, device)
    net.load_state_dict(load_saved_state(selected['checkpoint']))
    net.eval()
    synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        logits, nodes, _, features = net.forward_logits(adjacency, image_tensor)
    synchronize(device)
    elapsed = time.perf_counter() - start
    prediction = logits.argmax(1).cpu().numpy().reshape(test_gt.shape) + 1
    mask = test_gt > 0
    metrics = classification_metrics(test_gt[mask], prediction[mask], config['classes'])
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / 'predictions.npz', prediction=prediction,
                        test_indices=np.flatnonzero(mask), true_labels=test_gt[mask],
                        predicted_labels=prediction[mask], selected_stage2=nodes.cpu().numpy())
    np.savez_compressed(run_dir / 'test_features.npz', features=features[torch.as_tensor(mask.ravel(), device=device)].cpu().numpy(),
                        labels=test_gt[mask], flat_indices=np.flatnonzero(mask))
    result = dict(seed=selected['seed'], gamma=selected['gamma'], best_epoch=selected['best_epoch'],
                  validation_OA_percent=selected['best_val_oa'] * 100,
                  training_seconds=selected['training_seconds'], testing_seconds=elapsed,
                  total_seconds=selected['training_seconds'] + elapsed,
                  parameters=selected['parameters'], trainable_parameters=selected['trainable_parameters'], **metrics)
    write_json(run_dir / 'metrics.json', result)
    if plots:
        labeled_prediction = prediction.copy()
        # This is the union of split masks; no unknown reference classes are used.
        split_file = np.load(run_dir / 'splits.npz')
        labeled_prediction[(split_file['train_gt'] + split_file['val_gt'] + split_file['test_gt']) == 0] = 0
        Draw_Classification_Map(labeled_prediction, str(run_dir / 'classification'))
        plot_test_tsne(features.detach().cpu().numpy(), test_gt,
                       run_dir / 'test_tsne.png', selected['seed'], metrics['OA'])
    print(f'FINAL {config["dataset"]} seed={selected["seed"]} OA={metrics["OA"]:.2f}% '
          f'AA={metrics["AA"]:.2f}% kappa={metrics["Kappa"]:.2f}% F1={metrics["macro_F1"]:.2f}%', flush=True)
    del net, logits, features
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def source_hashes():
    base = Path(__file__).resolve().parent
    return {name: hashlib.sha256((base / name).read_bytes()).hexdigest()
            for name in ['main.py', 'model.py', 'slic.py', 'visual.py']}


def run_dataset(args, dataset, root_dir, device):
    config = dict(PAPER_CONFIG[dataset], dataset=dataset)
    for name in ['epochs', 'dim', 'scale']:
        override = getattr(args, name)
        if override is not None:
            config[name] = override
    image, gt = load_scene(args.data_root, dataset)
    config['bands'] = image.shape[-1]
    standardized = StandardScaler().fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape).astype(np.float32)
    out = Path(root_dir) / dataset
    out.mkdir(parents=True, exist_ok=True)
    gamma_values = sorted(set(args.gamma_values if args.gamma_values is not None else [args.gamma]))
    if any(not np.isfinite(gamma) or gamma <= 0 for gamma in gamma_values):
        raise ValueError('All gamma candidates must be finite and positive')
    write_json(out / 'experiment.json', dict(config=config, seeds=args.seeds, top_k=args.top_k,
               gamma_candidates=gamma_values, selection='highest validation OA; validation loss breaks ties',
               gamma_selection='mean best validation OA across all seeds; mean loss breaks ties',
               split_protocol='30/class; insufficient classes use 15; validation=5/class if remainder>5 else 1',
               train_per_class=args.train_samples, validation_only=args.validation_only,
               sd_ddof=1, device=str(device), source_sha256=source_hashes(),
               versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                             scipy=scipy.__version__, sklearn=sklearn.__version__, skimage=skimage.__version__)))
    splits = {}
    for seed in args.seeds:
        tr, te, va = get_Samples_GT(seed, gt, config['classes'], args.train_samples, 5, 'same_num')
        assert_disjoint_splits(gt, tr, va, te)
        splits[seed] = (tr, va, te)
        run_dir = out / f'seed_{seed}'
        run_dir.mkdir(exist_ok=True)
        np.savez_compressed(run_dir / 'splits.npz', train_gt=tr, val_gt=va, test_gt=te,
                            train_indices=np.flatnonzero(tr), val_indices=np.flatnonzero(va),
                            test_indices=np.flatnonzero(te))
        write_json(run_dir / 'split_counts.json', split_counts(tr, va, te, config['classes']))
    graph_start = time.perf_counter()
    # No training/validation/test labels are supplied to segmentation or graph construction.
    segmenter = slic.LDA_SLIC(standardized)
    Q, S, _, segments = segmenter.simple_superpixel_no_LDA(config['scale'], gamma_values[0])
    if args.top_k > Q.shape[1]:
        raise ValueError('top_k exceeds the realized number of superpixels')
    np.savez_compressed(out / 'graph.npz', segments=segments, node_spectra=S)
    image_tensor = torch.as_tensor(standardized, device=device)
    validation_results = []
    for gamma in gamma_values:
        adjacency_np = slic.gaussian_affinity_graph(segments, S, gamma)
        adjacency = torch.as_tensor(adjacency_np, device=device)
        graph_stats = dict(gamma=gamma, nodes=len(S), undirected_edges=int(np.count_nonzero(adjacency_np) // 2),
                           diagonal_zero=bool(np.all(np.diag(adjacency_np) == 0)),
                           symmetric=bool(np.array_equal(adjacency_np, adjacency_np.T)))
        write_json(out / f'graph_gamma_{gamma}.json', graph_stats)
        for seed in args.seeds:
            tr, va, _ = splits[seed]
            record = train_candidate(image_tensor, adjacency, Q, segments, tr, va,
                        config, seed, gamma, args.top_k,
                        out / f'gamma_{gamma}' / f'seed_{seed}', device)
            validation_results.append(record)
    chosen_gamma = select_gamma(validation_results)
    write_json(out / 'selection.json', dict(selected_gamma=chosen_gamma, candidates=validation_results,
               test_used_for_selection=False, segmentation_and_search_seconds=time.perf_counter() - graph_start))
    print(f'{dataset}: selected gamma={chosen_gamma:g} using validation only', flush=True)
    if args.validation_only:
        return
    adjacency = torch.as_tensor(slic.gaussian_affinity_graph(segments, S, chosen_gamma), device=device)
    test_results = []
    for seed in args.seeds:
        chosen = next(r for r in validation_results if r['seed'] == seed and r['gamma'] == chosen_gamma)
        tr, _, te = splits[seed]
        test_results.append(evaluate_selected(image_tensor, adjacency, Q, segments, tr, te,
                           config, chosen, args.top_k, out / f'seed_{seed}', device, not args.no_plots))
    keys = ['OA', 'AA', 'Kappa', 'macro_F1', 'per_class_recall']
    summary = dict(dataset=dataset, runs=len(test_results), selected_gamma=chosen_gamma,
                   units='percent', standard_deviation='sample (ddof=1)', results=test_results)
    for key in keys:
        values = np.asarray([record[key] for record in test_results])
        summary[key] = dict(mean=values.mean(axis=0),
                            sd=values.std(axis=0, ddof=1) if len(values) > 1 else None)
    write_json(out / 'summary.json', summary)
    with (out / 'summary.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['metric', 'mean_percent', 'sample_sd_percent'])
        for key in keys[:-1]:
            writer.writerow([key, summary[key]['mean'], summary[key]['sd']])
        for i in range(config['classes']):
            sd = summary['per_class_recall']['sd']
            writer.writerow([f'class_{i + 1}', summary['per_class_recall']['mean'][i], None if sd is None else sd[i]])


def parse_args(argv=None):
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=list(PAPER_CONFIG), default=['IP'])
    parser.add_argument('--data-root', type=Path, default=base / 'datasets')
    parser.add_argument('--output-root', type=Path, default=base / 'results')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument('--train-samples', type=int, default=30)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--dim', type=int)
    parser.add_argument('--scale', type=float)
    parser.add_argument('--top-k', type=int, default=5)
    gamma_group = parser.add_mutually_exclusive_group()
    gamma_group.add_argument('--gamma', type=float, default=1.0,
                             help='Fixed gamma; 1.0 is a starting value, not a recovered experimental setting')
    gamma_group.add_argument('--gamma-values', type=float, nargs='+',
                             help='Select one gamma per dataset using validation runs only')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--validation-only', action='store_true',
                        help='For tuning hidden dimension/scale/k: never calculate test scores')
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('seeds must be unique')
    if args.epochs is not None and args.epochs < 1:
        parser.error('epochs must be positive')
    if args.dim is not None and args.dim < 2:
        parser.error('dim must be at least two')
    if args.top_k < 1 or args.train_samples < 1 or args.threads < 1:
        parser.error('top-k, train-samples and threads must be positive')
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    root = args.output_root / ('paper_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    root.mkdir(parents=True, exist_ok=False)
    print('Outputs:', root.resolve(), flush=True)
    for dataset in args.datasets:
        run_dataset(args, dataset, root, device)


if __name__ == '__main__':
    main()
