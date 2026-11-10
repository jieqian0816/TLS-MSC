"""TLS-MSC: complete graph propagation and two selective local-refinement stages.

Matches the manuscript feature path, including detached ten-bin feature entropy,
fixed initialized global projections, and unweighted TLS-normalized graph features.
"""
import numpy as np
from scipy.stats import entropy
import torch
from torch import nn
from torch.nn import functional as F


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.BN = nn.BatchNorm1d(in_dim)
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = nn.LeakyReLU()

    def A_to_D_inv(self, adjacency):
        return torch.diag(adjacency.sum(1).rsqrt())

    def forward(self, g, h):
        h = self.BN(h)
        with_loops = g + torch.eye(g.shape[0], device=g.device, dtype=g.dtype)
        inv = with_loops.sum(1).rsqrt()
        normalized = inv[:, None] * with_loops * inv[None, :]
        return self.act(self.proj(normalized @ h))


def superpixel_labels(train_samples_gt, segments):
    """Eq. (13): only training labels; empty/conflicting regions map to zero."""
    labels = train_samples_gt.reshape(-1).to(device=segments.device, dtype=torch.long)
    ids = segments.reshape(-1).long()
    region_ids = torch.unique(ids)
    if not torch.equal(region_ids, torch.arange(len(region_ids), device=ids.device)):
        raise ValueError('Superpixel IDs must be contiguous and zero-based')
    result = torch.zeros(len(region_ids), dtype=torch.long, device=segments.device)
    for i in range(len(region_ids)):
        observed = torch.unique(labels[ids == i])
        observed = observed[observed > 0]
        if observed.numel() == 1:
            result[i] = observed[0]
    return result


def label_scores(adj_matrix, node_labels, superpixel_features,
                 alpha=0.6, use_entropy=True, bins=10):
    """Eqs. (14)--(16); unresolved state zero participates in state diversity."""
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be in [0,1]')
    boundary = torch.zeros(adj_matrix.shape[0], device=adj_matrix.device,
                           dtype=superpixel_features.dtype)
    for i in range(adj_matrix.shape[0]):
        neighbors = torch.nonzero(adj_matrix[i] > 0, as_tuple=False).flatten()
        boundary[i] = float(torch.unique(node_labels[neighbors]).numel() > 1)
    if use_entropy:
        rows = superpixel_features.detach().cpu().numpy()
        values = []
        for row in rows:
            hist = np.histogram(row, bins=bins)[0]
            values.append(entropy(hist / hist.sum() + 1e-10))
        heterogeneity = torch.as_tensor(values, device=adj_matrix.device,
                                        dtype=superpixel_features.dtype)
    else:
        # Explicit alternative for external ablation scripts, not the paper default.
        heterogeneity = superpixel_features.var(dim=1, unbiased=False)
    normalized = (heterogeneity - heterogeneity.min()) / (
        heterogeneity.max() - heterogeneity.min() + 1e-10)
    return alpha * boundary + (1 - alpha) * normalized


def top_k(scores, g, h, k):
    """Select exactly k nodes; do not pool/rewire g or multiply h by scores.

    Five return values preserve the original helper's public interface.
    The third and fourth values both identify the selected nodes.
    """
    if isinstance(k, bool) or int(k) != k or not 1 <= int(k) <= g.shape[0]:
        raise ValueError('k must be an integer in [1, number of nodes]')
    idx = torch.topk(scores, int(k)).indices
    selected = torch.zeros(scores.numel(), dtype=torch.bool, device=scores.device)
    selected[idx] = True
    remaining = torch.arange(scores.numel(), device=scores.device)[~selected]
    return g, h, idx, idx, remaining[:int(k)]


class Topo_Label_Score(nn.Module):
    def __init__(self, k, in_dim, node_labels):
        super().__init__()
        self.k = k
        self.BN = nn.BatchNorm1d(in_dim)
        self.proj = nn.Linear(in_dim, 1)
        # Hard index routing gives this projection no classification gradient.
        self.proj.requires_grad_(False)
        self.register_buffer('node_labels', node_labels.clone())

    def scores(self, g, normalized_features):
        degree = g.sum(1)
        # Zero rows remain zero for an isolated node (no artificial TLS self-loop).
        inv = torch.zeros_like(degree)
        nonzero = degree > 0
        inv[nonzero] = degree[nonzero].reciprocal()
        neighbors = inv[:, None] * (g @ normalized_features)
        local = (normalized_features - neighbors).abs().sum(1) + degree.sigmoid()
        global_score = self.proj(neighbors).squeeze(-1).softmax(dim=0)
        label = label_scores(g, self.node_labels, normalized_features,
                             alpha=0.6, use_entropy=True, bins=10)
        return torch.sigmoid((local + global_score) * label)

    def forward(self, g, h):
        normalized_features = self.BN(h)
        return top_k(self.scores(g, normalized_features), g, normalized_features, self.k)


def extract_superpixel_regions(data, superpixel_indices, mask_matrix):
    regions, information = [], []
    for index in superpixel_indices:
        coords = torch.nonzero(mask_matrix == index, as_tuple=False)
        if not coords.numel():
            raise ValueError('Selected superpixel has no pixels')
        lo = coords.amin(dim=0)
        hi = coords.amax(dim=0) + 1
        r0, c0 = lo.tolist()
        r1, c1 = hi.tolist()
        region = data[r0:r1, c0:c1, :]
        regions.append(region)
        information.append((coords, r0, c0))
    return regions, information


def write_back_features(original_data, results, processed_features):
    """Eq. (21): write only within selected superpixel supports, not full crops."""
    modified = original_data.clone()
    for (coords, r0, c0), features in zip(results, processed_features):
        rows, cols = coords[:, 0], coords[:, 1]
        modified[rows, cols] = features[rows - r0, cols - c0]
    return modified


class BoundaryNodeCNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.BatchNorm2d(in_dim), nn.Conv2d(in_dim, 64, 3, padding=1), nn.LeakyReLU(),
            nn.BatchNorm2d(64), nn.Conv2d(64, out_dim, 3, padding=1), nn.LeakyReLU())

    def forward(self, data, boundary_nodes, gcn_h, segments_mask):
        regions, information = extract_superpixel_regions(data, boundary_nodes, segments_mask)
        processed = []
        for region in regions:
            patch = region.permute(2, 0, 1).unsqueeze(0)
            features = self.cnn(patch).sigmoid().squeeze(0).permute(1, 2, 0)
            processed.append(features)
        return processed, information


Mixed_Superpixel_CNN = BoundaryNodeCNN


class SSConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5):
        super().__init__()
        self.BN = nn.BatchNorm2d(in_ch)
        self.point_conv = nn.Conv2d(in_ch, out_ch // 2, 1, bias=False)
        self.depth_conv = nn.Conv2d(out_ch // 2, out_ch, kernel_size,
                                    padding=kernel_size // 2)
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()

    def forward(self, x):
        return self.Act2(self.depth_conv(self.Act1(self.point_conv(self.BN(x)))))


class GCN_in_CNN(nn.Module):
    def __init__(self, height, width, in_dim, dim, n_classes, Q, segments,
                 train_gt, top_k=5):
        super().__init__()
        self.height, self.width = height, width
        self.n_classes = n_classes
        if dim < 2:
            raise ValueError('Hidden dimension must be at least two')
        if not 1 <= int(top_k) <= Q.shape[1]:
            raise ValueError('top_k exceeds the number of superpixels')
        self.register_buffer('Q', Q.float(), persistent=False)
        self.register_buffer('norm_col_Q', Q / Q.sum(0, keepdim=True), persistent=False)
        self.register_buffer('segments', segments.long(), persistent=False)
        node_labels = superpixel_labels(train_gt, segments)
        self.CNN_denoise = nn.Sequential()
        for stage, channels in [(1, in_dim), (2, dim)]:
            self.CNN_denoise.add_module('CNN_denoise_BN' + str(stage), nn.BatchNorm2d(channels))
            self.CNN_denoise.add_module('CNN_denoise_Conv' + str(stage), nn.Conv2d(channels, dim, 1))
            self.CNN_denoise.add_module('CNN_denoise_Act' + str(stage), nn.LeakyReLU())
        self.gcn1, self.gcn2, self.gcn3 = (GCN(dim, dim) for _ in range(3))
        self.pool1 = Topo_Label_Score(top_k, dim, node_labels)
        self.pool2 = Topo_Label_Score(top_k, dim, node_labels)
        self.NodeCNN1, self.NodeCNN2 = BoundaryNodeCNN(dim, dim), BoundaryNodeCNN(dim, dim)
        self.conv1 = SSConv(dim, dim, kernel_size=5)
        self.classifier = nn.Linear(dim, n_classes)

    def forward_logits(self, g, x, q=None):
        """Logits for numerically stable summed cross-entropy on training pixels."""
        h, w, _ = x.shape
        clean = self.CNN_denoise(x.permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0)
        initial = self.norm_col_Q.t() @ clean.reshape(h * w, -1)
        g1 = self.gcn1(g, initial)
        _, u1, _, selected1, _ = self.pool1(g, g1)
        patches1, info1 = self.NodeCNN1(clean, selected1, u1, self.segments)
        g2 = self.gcn2(g, u1)
        _, u2, _, selected2, remaining2 = self.pool2(g, g2)
        patches2, info2 = self.NodeCNN2(clean, selected2, u2, self.segments)
        g3 = self.gcn3(g, u2)
        d1 = (self.Q @ g2).reshape(h, w, -1).sigmoid()
        d2 = (self.Q @ g3).reshape(h, w, -1).sigmoid()
        fused = clean + write_back_features(d1, info1, patches1) + write_back_features(d2, info2, patches2)
        features = self.conv1(fused.permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0)
        features = features.reshape(h * w, -1)
        return self.classifier(features), selected2, remaining2, features

    def forward(self, g, x, q=None):
        logits, selected, remaining, features = self.forward_logits(g, x, q)
        return logits.softmax(dim=-1), selected, remaining, features


TLS_MSC = GCN_in_CNN
