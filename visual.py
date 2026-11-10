import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pylab
import random
from matplotlib import cm
from sklearn import metrics
import time
from sklearn import preprocessing
import torch
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from skimage.segmentation import mark_boundaries
from matplotlib.colors import ListedColormap

def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX


def get_Samples_GT(seed, gt, class_count, train_ratio, val_ratio=5, samples_type='same_num'):
    """Return (train, test, validation), preserving the original return order.

    Paper path: same_num, 30 training/class; classes too small to retain validation
    and test use at most 15 training pixels (IP classes 7 and 9). From the remainder,
    reserve five validation pixels if more than five remain, otherwise one.
    Every other labeled pixel is test-only. Background zero is never sampled.
    A local RNG makes splits reproducible without changing global random state.
    """
    gt = np.asarray(gt)
    if gt.ndim != 2 or not np.isfinite(gt).all() or not np.equal(gt, np.floor(gt)).all():
        raise ValueError('gt must be a finite two-dimensional integer label image')
    if gt.min() < 0 or gt.max() > class_count:
        raise ValueError('Labels must be in 0,...,class_count')
    if samples_type not in {'same_num', 'ratio'}:
        raise ValueError('samples_type must be same_num or ratio')
    if samples_type == 'same_num':
        if int(train_ratio) != train_ratio or train_ratio < 1:
            raise ValueError('same_num requires a positive integer training count')
        if val_ratio != 5:
            raise ValueError('Paper protocol requires val_ratio=5 (five validation pixels/class)')
    elif not (0 < train_ratio < 1 and 0 < val_ratio < 1):
        raise ValueError('ratio mode requires ratios in (0,1); it is not the paper protocol')
    rng = np.random.default_rng(seed)
    flat = gt.ravel().astype(np.int64)
    train, val, test = (np.zeros_like(flat) for _ in range(3))
    for cls in range(1, class_count + 1):
        indices = np.flatnonzero(flat == cls)
        if len(indices) < 3:
            raise ValueError(f'Class {cls} needs at least 3 pixels for disjoint train/val/test')
        if samples_type == 'same_num':
            n_train = int(train_ratio)
            if len(indices) - n_train < 2:
                n_train = min(15, n_train, len(indices) - 2)
            remaining = len(indices) - n_train
            n_val = 5 if remaining > 5 else 1
        else:
            n_train = min(max(1, int(np.ceil(len(indices) * train_ratio))), len(indices) - 2)
            n_val = min(max(1, int(np.ceil(len(indices) * val_ratio))), len(indices) - n_train - 1)
        shuffled = rng.permutation(indices)
        train_idx = shuffled[:n_train]
        val_idx = shuffled[n_train:n_train + n_val]
        test_idx = shuffled[n_train + n_val:]
        train[train_idx], val[val_idx], test[test_idx] = cls, cls, cls
    train, val, test = (a.reshape(gt.shape) for a in (train, val, test))
    assert_disjoint_splits(gt, train, val, test)
    return train, test, val


def assert_disjoint_splits(gt, train, validation, test):
    """Fail before training if masks overlap, lose pixels, or alter labels."""
    gt = np.asarray(gt)
    masks = []
    for name, split in [('train', train), ('validation', validation), ('test', test)]:
        split = np.asarray(split)
        if split.shape != gt.shape:
            raise ValueError(f'{name} mask shape differs from ground truth')
        mask = split > 0
        if not np.any(mask) or not np.array_equal(split[mask], gt[mask]):
            raise ValueError(f'{name} mask is empty or changes reference labels')
        if np.any(split[~mask] != 0):
            raise ValueError(f'{name} mask has negative labels')
        masks.append(mask.astype(np.uint8))
    coverage = sum(masks)
    if np.any(coverage > 1):
        raise ValueError('Training, validation and test masks overlap')
    if not np.array_equal(coverage > 0, gt > 0):
        raise ValueError('Splits must partition all labeled pixels and exclude background')


def split_counts(train, validation, test, class_count):
    return {name: [int(np.count_nonzero(np.asarray(split) == cls))
                   for cls in range(1, class_count + 1)]
            for name, split in [('training', train), ('validation', validation), ('testing', test)]}


def classification_metrics(y_true, y_pred, class_count):
    """All metrics from ONE held-out confusion matrix; returned values are percent."""
    truth = np.asarray(y_true, dtype=np.int64).ravel()
    pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if truth.size == 0 or truth.shape != pred.shape:
        raise ValueError('Nonempty paired ground truth and predictions required')
    if truth.min() < 1 or truth.max() > class_count or pred.min() < 1 or pred.max() > class_count:
        raise ValueError('Metric inputs must use class IDs 1,...,class_count')
    confusion = np.bincount((truth - 1) * class_count + pred - 1,
                            minlength=class_count ** 2).reshape(class_count, class_count)
    support = confusion.sum(1)
    if np.any(support == 0):
        raise ValueError('Every reported class must have held-out test pixels')
    tp = np.diag(confusion).astype(np.float64)
    predicted = confusion.sum(0)
    recall = tp / support
    f1 = np.divide(2 * tp, support + predicted,
                    out=np.zeros_like(tp), where=(support + predicted) > 0)
    oa = tp.sum() / truth.size
    expected = float(np.dot(support.astype(float), predicted.astype(float)) / truth.size ** 2)
    kappa = (oa - expected) / (1 - expected) if expected < 1 else 0.0
    return dict(OA=100 * oa, AA=100 * recall.mean(), Kappa=100 * kappa,
                macro_F1=100 * f1.mean(), per_class_recall=100 * recall,
                confusion_matrix=confusion, test_support=support)


def plot_test_tsne(features, test_gt, path, seed, oa, max_per_class=500):
    """Use test pixels only. Legend distinguishes plotted n from test support N."""
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    gt = np.asarray(test_gt).ravel()
    features = np.asarray(features)[gt > 0]
    labels = gt[gt > 0]
    rng = np.random.default_rng(seed)
    chosen = []
    for cls in np.unique(labels):
        available = np.flatnonzero(labels == cls)
        chosen.extend(rng.choice(available, min(max_per_class, len(available)), replace=False))
    chosen = np.asarray(chosen)
    if chosen.size < 3:
        return
    embedding = TSNE(n_components=2, perplexity=min(30, chosen.size - 1),
                     learning_rate=200, random_state=seed).fit_transform(
                         StandardScaler().fit_transform(features[chosen]))
    fig, ax = plt.subplots(figsize=(12, 10))
    classes = np.unique(labels)
    palette = plt.get_cmap('tab20', len(classes))
    for i, cls in enumerate(classes):
        selected = labels[chosen] == cls
        count = int(selected.sum())
        total = int(np.count_nonzero(labels == cls))
        ax.scatter(embedding[selected, 0], embedding[selected, 1], s=15, color=palette(i),
                   alpha=0.7, label=f'Class {cls} (n={count}; N_test={total})')
    ax.set_title(f'Seed {seed}; held-out OA={oa:.2f}%')
    ax.axis('off')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.savefig(path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)




def GT_To_One_Hot(gt, class_count):
    
    GT_One_Hot = []  
    [height, width]=gt.shape
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            temp = np.zeros(class_count,dtype=np.float32)
            if gt[i, j] != 0:
                temp[int( gt[i, j]) - 1] = 1
            GT_One_Hot.append(temp)
    GT_One_Hot = np.reshape(GT_One_Hot, [height, width, class_count])
    return GT_One_Hot


def Draw_Classification_Map(label, name: str, scale: float = 4.0, dpi: int = 400, segments=None, top5_masks=None, adj_masks=None, top5_masks_deg=None):

        import spectral as spy
        fig, ax = plt.subplots()
        numlabel = np.array(label)
        # 如果需要对图像进行进一步操作（例如设置标题、调整颜色条等），可以使用 v
        v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number, cmap='jet')

        # 超像素分割叠加边界
        if segments is not None:
            for mk in segments:
                plt.contour(mk, colors='white', linewidths=0.15)

        # 标注超像素的邻居
        if adj_masks is not None:
            for mask2 in adj_masks:
                plt.contour(mask2, colors='magenta', linewidths=0.15)

        # 标注评分最高的 5 个超像素区域
        if top5_masks is not None:
            for mask in top5_masks:
                plt.contour(mask, colors=(1, 1, 0), linewidths=0.15)

        # 标注最不重要的 5 个超像素区域
        if top5_masks_deg is not None:
            for mask in top5_masks_deg:
                plt.contour(mask, colors='green', linewidths=0.15)

        ax.set_axis_off()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
        # fig.set_size_inches(numlabel.shape[1] * scale / dpi, numlabel.shape[0] * scale / dpi)
        foo_fig = plt.gcf()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # name='suiji2'

        foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
        plt.close(fig)
        pass

# def Draw_Classification_Map(label, name: str, scale: float = 4.0, dpi: int = 400, segments=None, top5_masks=None,
#                             adj_masks=None, top5_masks_deg=None):
#     fig, ax = plt.subplots()
#     numlabel = np.array(label)
#
#     # 使用 matplotlib 的 imshow 替代 spectral 的 imshow
#     v = ax.imshow(numlabel.astype(np.int16), cmap='jet')
#
#     # 添加颜色条
#     cbar = fig.colorbar(v, ax=ax, fraction=0.046, pad=0.04)
#     cbar.set_label('Class Labels', rotation=270, labelpad=15)
#
#     # 超像素分割叠加边界
#     if segments is not None:
#         for mk in segments:
#             plt.contour(mk, colors='white', linewidths=0.15)
#
#     # 标注超像素的邻居
#     if adj_masks is not None:
#         for mask2 in adj_masks:
#             plt.contour(mask2, colors='magenta', linewidths=0.15)
#
#     # 标注评分最高的5个超像素区域
#     if top5_masks is not None:
#         for mask in top5_masks:
#             plt.contour(mask, colors=(1, 1, 0), linewidths=0.15)
#
#     # 标注最不重要的5个超像素区域
#     if top5_masks_deg is not None:
#         for mask in top5_masks_deg:
#             plt.contour(mask, colors='green', linewidths=0.15)
#
#     ax.set_axis_off()
#     ax.xaxis.set_visible(False)
#     ax.yaxis.set_visible(False)
#     fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
#
#     # 调整布局以容纳颜色条
#     plt.subplots_adjust(top=1, bottom=0, right=0.85, left=0, hspace=0, wspace=0)
#
#     foo_fig = plt.gcf()
#     plt.gca().xaxis.set_major_locator(plt.NullLocator())
#     plt.gca().yaxis.set_major_locator(plt.NullLocator())
#
#     foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0, bbox_inches='tight')
#     plt.close(fig)


# numpy型的处理
# def count_connected_components(adj_matrix):
#     """
#     处理邻接矩阵，返回连通数
#     """
#     n = len(adj_matrix)  # 图的顶点数
#     visited = [False] * n  # 记录顶点是否被访问过
#     connected_count = 0  # 连通分量的数量
#     def dfs(u):
#         visited[u] = True
#         for v in range(n):
#             if adj_matrix[u][v] and not visited[v]:  # 如果u和v之间有边且v未被访问过
#                 dfs(v)
#
#     # 遍历所有顶点
#     for u in range(n):
#         if not visited[u]:  # 如果u未被访问过，说明它是一个新的连通分量的起点
#             connected_count += 1
#             dfs(u)  # 遍历u所在的连通分量
#
#     return connected_count
#
#
# def process_subgraphs(adj_matrix, node_sequences):
#     """
#     根据给定的节点序列生成子图邻接矩阵、环数和子图对象
#     参数:
#         adj_matrix (np.array): 原始邻接矩阵 [n x n]
#         node_sequences (list): 包含多个节点序列的，例如 [2,3,4]
#
#     返回:
#         sub_adjs (list): 子图邻接矩阵列表
#         neighbor_ids(list): 目标节点的邻居标号列表
#         cycle_counts (list): 各子图环数列表
#         sub_graphs (list): 子图NetworkX对象列表
#     """
#     sub_adjs = []
#     neighbor_ids =[]
#     cycle_counts = []
#     sub_graphs = []
#
#     # 验证输入有效性
#     node_sequences = np.unique(node_sequences) # 去重
#     if len(node_sequences) == 0:
#         raise ValueError("Node sequence cannot be empty")
#     if np.max(node_sequences) >= adj_matrix.shape[0]:
#         raise IndexError("Node index out of adjacency matrix bounds")
#
#     # 遍历每个节点
#     for node in node_sequences:
#         # --- 提取子图邻接矩阵 ---
#         neighbors = np.append(np.where(adj_matrix[node] != 0)[0], node)  # 获取目标节点及其邻居
#         neighbor_ids.append(np.setdiff1d(neighbors, [node]))  # 使用 np.setdiff1d 删除指定值
#         neighbors = np.unique(neighbors)  # 去重排序
#         sub_adj = adj_matrix[neighbors][:, neighbors]  # 提取子图邻接矩阵（保留所有边）
#         sub_adjs.append(sub_adj)
#
#         # 计算环数（贝蒂数1）
#         edge_count = np.sum(sub_adj) // 2  # 图边数
#         node_count = len(neighbors)   # 节点数
#         components = count_connected_components(sub_adj)  # 连通数
#         beta_1 = edge_count - node_count + components  # 环的数量=边-节点+连通数
#         cycle_counts.append(beta_1)
#
#         # 构建networkx图对象
#         G_sub = nx.from_numpy_array(sub_adj)
#         # 保持原始节点标签（重要：将子图节点映射回原编号）
#         mapping = {i: orig for i, orig in enumerate(neighbors)}
#         G_sub = nx.relabel_nodes(G_sub, mapping)
#         sub_graphs.append(G_sub)
#
#     return sub_adjs, neighbor_ids, cycle_counts, sub_graphs


# def visualize_results(G_full, sub_graphs, cycle_counts, node_sequences):
#     """
#     可视化全图及所有子图（自动分两行布局，完全隐藏坐标轴边框）
#     """
#     n = len(sub_graphs)
#     n_cols = (n + 1) // 2  # 每行子图数
#     fig, axs = plt.subplots(2, n_cols + 1,
#                             figsize=(3 * (n_cols + 1), 8),
#                             gridspec_kw={'width_ratios': [1] + [1] * n_cols})
#
#     # --- 初始化：强制隐藏所有坐标轴 ---
#     for row in range(2):
#         for col in range(n_cols + 1):
#             axs[row, col].axis('off')  # 全部关闭坐标轴
#
#     # --- 绘制全图（单独激活并设置）---
#     axs[0, 0].axis('on')  # 仅全图坐标轴保持开启
#     pos_full = nx.spring_layout(G_full)
#     nx.draw(G_full, pos_full, ax=axs[0, 0], with_labels=True,
#             node_color='skyblue', edge_color='grey', node_size=10)
#     axs[0, 0].set_title("Original Graph", fontsize=10)
#     axs[0, 0].set_xticks([])  # 隐藏刻度
#     axs[0, 0].set_yticks([])
#     axs[0, 0].spines['top'].set_visible(False)  # 隐藏边框
#     axs[0, 0].spines['right'].set_visible(False)
#     axs[0, 0].spines['bottom'].set_visible(False)
#     axs[0, 0].spines['left'].set_visible(False)
#
#     # --- 动态填充子图 ---
#     for idx, (G_sub, beta) in enumerate(zip(sub_graphs, cycle_counts)):
#         row = 1 if idx >= n_cols else 0
#         col = (idx % n_cols) + 1  # 从第2列开始
#
#         # 绘制子图（保持坐标轴关闭）
#         pos = nx.spring_layout(G_sub)
#         nx.draw(G_sub, pos, ax=axs[row, col], with_labels=True,
#                 node_color='#FFD700' if row == 0 else '#90EE90',
#                 edge_color='#444444', node_size=10)
#         axs[row,col].set_title(f"Node {node_sequences[idx]}'s Subgraph\nCycles: {beta}")
#
#         # 强制隐藏所有子图的坐标元素
#         axs[row, col].set_xticks([])
#         axs[row, col].set_yticks([])
#         for spine in axs[row, col].spines.values():
#             spine.set_visible(False)
#
#     plt.tight_layout(pad=1.2)
#     plt.show()
#     plt.pause(10)  # 暂停 10 秒，然后继续运行代码
#
#
# def split_heterogeneous_nodes(A, H, Q, orig_data):
#     """
#     输入:
#         A: 初始邻接矩阵 (n_nodes, n_nodes)
#         H: 节点特征矩阵 (n_nodes, n_features)
#         Q: 像素-超像素关系矩阵 (n_pixels, n_nodes)
#         orig_data: 原始输入的data_pca(n_pixels, n_features最大最小化了)
#     输出:
#         new_A: 裂变后的邻接矩阵
#         new_H: 裂变后的特征矩阵
#         new_Q: 裂变后的关系矩阵
#     """
#     np.fill_diagonal(A, 0)  # 去除自邻接
#     # 拷贝原始数据避免污染
#     new_A = A.copy()
#     new_H = H.copy()
#     new_Q = Q.copy()
#     n_orig_nodes = A.shape[0]
#
#     # 计算节点度数并选出度最大的节点
#     degrees = np.sum(A, axis=1)
#     max_degree = np.max(degrees)
#     max_degree_nodes = np.where(degrees == max_degree)[0]
#
#     # 调用process_subgraphs函数获取需要裂变节点的局部邻接矩阵、局部环数、子图对象
#     sub_adjs, nei, cyc_counts, sub_graphs = process_subgraphs(A, max_degree_nodes)
#     G = nx.from_numpy_array(A)
#     # visualize_results(G, sub_graphs, cyc_counts, max_degree_nodes)
#
#     # node是取遍初始图中的度最大节点标号，cyc_i是node节点的局部环数
#     for node, cyc_i, original_neighbors in zip(max_degree_nodes, cyc_counts, nei):
#         cyc_i = int(cyc_i)
#         # 获取node超像素节点内所有像素的光谱数据
#         pixel_indices = np.where(Q[:, node] == 1)[0]
#         orig_pixel_feature = orig_data[pixel_indices, :]
#
#         n_samples = orig_pixel_feature.shape[0]  # 获取样本量
#         if n_samples < cyc_i:
#             # 如果样本量小于聚类簇数，将聚类簇数设置为样本量的三分之一并向下取整
#             # cyc_i = max(1, int(cyc_i / 20))  # 至少为 1，避免聚类簇数为 0
#             cyc_i = 5
#         else:
#             # cyc_i = max(1, int(cyc_i / 10))
#             cyc_i = 3
#
#         # 光谱K-means聚类
#         kmeans = KMeans(n_clusters=cyc_i)
#         labels = kmeans.fit_predict(orig_pixel_feature) # labels是每个像素所属的聚类编号
#
#         # 生成每个子节点特征 (均值)
#         sub_features = []
#         for cluster_id in range(cyc_i):
#             selected_indices = np.where(labels == cluster_id)[0]
#             selected_pixel_feature = orig_pixel_feature[selected_indices]
#             sub_features.append(selected_pixel_feature.mean(axis=0))
#
#         sub_features = np.array(sub_features)
#
#         # 更新特征矩阵H: 添加子节点及特征
#         new_H = np.vstack([new_H, sub_features])
#         # original_node_idx = node
#         # new_H = np.delete(new_H, original_node_idx, axis=0)  # 移除原节点
#
#         # 更新关系矩阵Q: 分配像素到新子节点
#         new_sub_cols = np.zeros((new_Q.shape[0], cyc_i))
#         for cluster_id in range(cyc_i):
#             sel_indices = pixel_indices[labels == cluster_id]
#             new_sub_cols[sel_indices, cluster_id] = 1
#
#         new_Q = np.hstack([new_Q, new_sub_cols])  # 添加新列
#         # new_Q = np.delete(new_Q, original_node_idx, axis=1)
#
#         # 更新邻接矩阵A: 动态调整矩阵
#         # 步骤1: 添加cyc_i个新节点占位
#         n_current = new_A.shape[0]
#         new_nodes = list(range(n_current, n_current + cyc_i))
#         new_A = np.pad(new_A, ((0, cyc_i), (0, cyc_i)), mode='constant')  # mode='constant'表示填充常数，默认为0
#
#         # 步骤2: 子节点间全连接
#         for i in new_nodes:
#             for j in new_nodes:
#                 if i != j:
#                     new_A[i, j] = 1
#                     new_A[j, i] = 1
#
#         # 步骤3: 连接子节点与外部邻居
#         for sub_node in new_nodes:
#             # 计算与每个邻居的相似度
#             candidate_features = new_H[original_neighbors]
#             similarities = cosine_similarity(
#                 new_H[sub_node].reshape(1, -1),
#                 candidate_features
#             )
#             # 找到最相似节点的索引（在candidate_features中的位置）
#             max_idx = np.argmax(similarities)
#             # 获取原始图中的节点索引
#             most_similar_node = original_neighbors[max_idx]
#             # 添加新连接（对称赋值）
#             new_A[sub_node, most_similar_node] = 1
#             new_A[most_similar_node, sub_node] = 1
#
#     # 删除所有裂变节点
#     original_node_idx = max_degree_nodes
#     new_H = np.delete(new_H, original_node_idx, axis=0)  # 移除原节点
#     new_Q = np.delete(new_Q, original_node_idx, axis=1)
#     new_A = np.delete(new_A, original_node_idx, axis=0)
#     new_A = np.delete(new_A, original_node_idx, axis=1)
#
#     return new_A, new_H, new_Q

# torch型的处理
def count_connected_components(adj_matrix):
    """
    处理邻接矩阵，返回连通数
    参数:
        adj_matrix (torch.Tensor): 邻接矩阵 [n x n]
    返回:
        connected_count (int): 连通分量数量
    """
    n = adj_matrix.shape[0]  # 图的顶点数
    visited = [False] * n  # 记录顶点是否被访问过
    connected_count = 0  # 连通分量的数量

    def dfs(u):
        visited[u] = True
        for v in range(n):
            if adj_matrix[u][v] and not visited[v]:  # 如果u和v之间有边且v未被访问过
                dfs(v)

    # 遍历所有顶点
    for u in range(n):
        if not visited[u]:  # 如果u未被访问过，说明它是一个新的连通分量的起点
            connected_count += 1
            dfs(u)  # 遍历u所在的连通分量

    return connected_count


def process_subgraphs(adj_matrix, node_sequences):
    """
    根据给定的节点序列生成子图邻接矩阵、环数和子图对象
    参数:
        adj_matrix (torch.Tensor): 原始邻接矩阵 [n x n]
        node_sequences (torch.Tensor): 包含多个节点序列的，例如 [2,3,4]
    返回:
        sub_adjs (list): 子图邻接矩阵列表
        neighbor_ids (list): 目标节点的邻居标号列表
        cycle_counts (list): 各子图环数列表
        sub_graphs (list): 子图NetworkX对象列表
    """
    sub_adjs = []
    neighbor_ids = []
    cycle_counts = []
    sub_graphs = []

    # 获取设备信息
    device = adj_matrix.device

    # 验证输入有效性
    node_sequences = torch.unique(node_sequences)  # 去重
    if len(node_sequences) == 0:
        raise ValueError("Node sequence cannot be empty")
    if torch.max(node_sequences) >= adj_matrix.shape[0]:
        raise IndexError("Node index out of adjacency matrix bounds")

    # 遍历每个节点
    for node in node_sequences:
        node = node.item()  # 将 torch.Tensor 转换为 Python int
        # --- 提取子图邻接矩阵 ---
        neighbors = torch.cat(
            [torch.where(adj_matrix[node] != 0)[0], torch.tensor([node], device=device)]
        )  # 获取目标节点及其邻居
        neighbor_ids.append(
            torch.tensor([n.item() for n in neighbors if n != node], device=device)  # 使用列表推导删除指定值
        )
        neighbors = torch.unique(neighbors)  # 去重排序
        sub_adj = adj_matrix[neighbors][:, neighbors]  # 提取子图邻接矩阵（保留所有边）
        sub_adjs.append(sub_adj)

        # 计算环数（贝蒂数1）
        edge_count = torch.sum(sub_adj) // 2  # 图边数
        node_count = len(neighbors)  # 节点数
        components = count_connected_components(sub_adj)  # 连通数
        beta_1 = edge_count - node_count + components  # 环的数量=边-节点+连通数
        cycle_counts.append(beta_1)

        # 构建networkx图对象
        G_sub = nx.from_numpy_array(sub_adj.cpu().numpy())  # 将子图邻接矩阵移动到CPU并转换为numpy数组
        # 保持原始节点标签（重要：将子图节点映射回原编号）
        mapping = {i: orig.item() for i, orig in enumerate(neighbors)}
        G_sub = nx.relabel_nodes(G_sub, mapping)
        sub_graphs.append(G_sub)

    return sub_adjs, neighbor_ids, cycle_counts, sub_graphs


def visualize_results(G_full, sub_graphs, cycle_counts, node_sequences):
    """
    可视化全图及所有子图（自动分两行布局，完全隐藏坐标轴边框）
    """
    n = len(sub_graphs)
    n_cols = (n + 1) // 2  # 每行子图数
    fig, axs = plt.subplots(2, n_cols + 1,
                            figsize=(3 * (n_cols + 1), 8),
                            gridspec_kw={'width_ratios': [1] + [1] * n_cols})

    # --- 初始化：强制隐藏所有坐标轴 ---
    for row in range(2):
        for col in range(n_cols + 1):
            axs[row, col].axis('off')  # 全部关闭坐标轴

    # --- 绘制全图（单独激活并设置）---
    axs[0, 0].axis('on')  # 仅全图坐标轴保持开启
    pos_full = nx.spring_layout(G_full)
    nx.draw(G_full, pos_full, ax=axs[0, 0], with_labels=True,
            node_color='skyblue', edge_color='grey', node_size=10)
    axs[0, 0].set_title("Original Graph", fontsize=10)
    axs[0, 0].set_xticks([])  # 隐藏刻度
    axs[0, 0].set_yticks([])
    axs[0, 0].spines['top'].set_visible(False)  # 隐藏边框
    axs[0, 0].spines['right'].set_visible(False)
    axs[0, 0].spines['bottom'].set_visible(False)
    axs[0, 0].spines['left'].set_visible(False)

    # --- 动态填充子图 ---
    for idx, (G_sub, beta) in enumerate(zip(sub_graphs, cycle_counts)):
        row = 1 if idx >= n_cols else 0
        col = (idx % n_cols) + 1  # 从第2列开始

        # 绘制子图（保持坐标轴关闭）
        pos = nx.spring_layout(G_sub)
        nx.draw(G_sub, pos, ax=axs[row, col], with_labels=True,
                node_color='#FFD700' if row == 0 else '#90EE90',
                edge_color='#444444', node_size=10)
        axs[row,col].set_title(f"Node {node_sequences[idx]}'s Subgraph\nCycles: {beta}")

        # 强制隐藏所有子图的坐标元素
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])
        for spine in axs[row, col].spines.values():
            spine.set_visible(False)

    plt.tight_layout(pad=1.2)
    plt.show()
    plt.pause(10)  # 暂停 10 秒，然后继续运行代码


def split_heterogeneous_nodes(A, H, Q, orig_data):
    """
    输入:
        A: 初始邻接矩阵 (n_nodes, n_nodes)
        H: 节点特征矩阵 (n_nodes, n_features)
        Q: 像素-超像素关系矩阵 (n_pixels, n_nodes)
        orig_data: 原始输入的data_pca(n_pixels, n_features最大最小化了)
    输出:
        new_A: 裂变后的邻接矩阵
        new_H: 裂变后的特征矩阵
        new_Q: 裂变后的关系矩阵
    """
    A.fill_diagonal_(0)  # 去除自邻接
    # 拷贝原始数据避免污染
    new_A = A.clone()
    new_H = H.clone()
    new_Q = Q.clone()
    n_orig_nodes = A.shape[0]

    # 计算节点度数并选出度最大的节点
    degrees = torch.sum(A, axis=1)
    max_degree = torch.max(degrees)
    max_degree_nodes = torch.where(degrees == max_degree)[0]

    # sorted_degrees, sorted_indices = torch.sort(degrees, descending=True)
    # print("节点度数的索引排名：")
    # for i, (degree, index) in enumerate(zip(sorted_degrees, sorted_indices)):
    #     print(f"第 {i + 1} 名: 索引 = {index.item()}, 度数 = {degree.item()}")

    # 调用process_subgraphs函数获取需要裂变节点的局部邻接矩阵、局部环数、子图对象
    sub_adjs, nei, cyc_counts, sub_graphs = process_subgraphs(A, max_degree_nodes)
    # G = nx.from_numpy_array(A.cpu().numpy())  # 如果需要可视化，可以取消注释
    # visualize_results(G, sub_graphs, cyc_counts, max_degree_nodes)

    # node是取遍初始图中的度最大节点标号，cyc_i是node节点的局部环数
    for node, cyc_i, original_neighbors in zip(max_degree_nodes, cyc_counts, nei):
        cyc_i = int(cyc_i)
        # 获取node超像素节点内所有像素的光谱数据
        pixel_indices = torch.where(Q[:, node] == 1)[0]
        orig_pixel_feature = orig_data[pixel_indices, :]

        n_samples = orig_pixel_feature.shape[0]  # 获取样本量
        if n_samples < cyc_i:
            # 如果样本量小于聚类簇数，将聚类簇数设置为样本量的三分之一并向下取整
            # cyc_i = max(1, int(cyc_i / 20))  # 至少为 1，避免聚类簇数为 0
            cyc_i = 3
        else:
            # cyc_i = max(1, int(cyc_i / 10))
            cyc_i = 3

        # 光谱K-means聚类
        kmeans = KMeans(n_clusters=3)
        labels = kmeans.fit_predict(orig_pixel_feature.cpu().detach().numpy())  # 分离梯度并转换为 numpy 数组

        # 生成每个子节点特征 (均值)
        # sub_features = []
        # for cluster_id in range(cyc_i):
        #     selected_indices = torch.where(torch.tensor(labels) == cluster_id)[0]
        #     selected_pixel_feature = orig_pixel_feature[selected_indices]
        #     sub_features.append(selected_pixel_feature.mean(axis=0))

        # sub_features = torch.stack(sub_features)  # 堆叠子节点的特征
        # mid_data = torch.matmul(new_Q, new_H)
        # all_features = torch.cat((mid_data, sub_features))
        # print('all_features.shape',all_features.shape)

        # 更新关系矩阵Q: 分配像素到新子节点
        new_sub_cols = torch.zeros((new_Q.shape[0], cyc_i), dtype=torch.float32, device=A.device)
        for cluster_id in range(cyc_i):
            sel_indices = pixel_indices[labels == cluster_id]
            new_sub_cols[sel_indices, cluster_id] = 1

        new_Q = torch.cat([new_Q, new_sub_cols], dim=1)  # 添加新列

        # 更新特征矩阵H: 添加子节点及特征
        norm_col_sub_Q = new_sub_cols / (torch.sum(new_sub_cols, 0, keepdim=True))  # 列归一化
        new_sub_H  = torch.mm(norm_col_sub_Q.t(), orig_data)
        new_H = torch.cat([new_H, new_sub_H], dim=0)
        # print('_______________')
        # print(new_H)
        # print('_______________')

        # 更新邻接矩阵A: 动态调整矩阵
        # 步骤1: 添加cyc_i个新节点占位
        n_current = new_A.shape[0]
        new_nodes = list(range(n_current, n_current + cyc_i))
        new_A = torch.nn.functional.pad(new_A, (0, cyc_i, 0, cyc_i), mode='constant', value=0)  # mode='constant'表示填充常数，默认为0

        # 步骤2: 子节点间全连接
        # for i in new_nodes:
        #     for j in new_nodes:
        #         if i != j:
        #             new_A[i, j] = 1
        #             new_A[j, i] = 1

        # 步骤3: 连接子节点与外部邻居
        for sub_node in new_nodes:
            # 计算与每个邻居的相似度
            candidate_features = new_H[original_neighbors]
            # print('candidate_features', candidate_features)
            # print('new_H', new_H)
            similarities = cosine_similarity(
                new_H[sub_node].unsqueeze(0).cpu().detach().numpy(),  # 分离梯度并转换为 numpy 数组
                candidate_features.cpu().detach().numpy()
            )
            # 找到最相似节点的索引（在candidate_features中的位置）
            max_idx = torch.argmax(torch.tensor(similarities))
            # 获取原始图中的节点索引
            most_similar_node = original_neighbors[max_idx]
            # 添加新连接（对称赋值）
            new_A[sub_node, most_similar_node] = 1
            new_A[most_similar_node, sub_node] = 1

    # 删除所有裂变节点
    original_node_idx = max_degree_nodes.tolist()  # 将张量转换为 Python 列表
    new_H = torch.cat([new_H[:original_node_idx[0]], new_H[original_node_idx[-1]+1:]], dim=0)  # 移除原节点
    new_Q = torch.cat([new_Q[:, :original_node_idx[0]], new_Q[:, original_node_idx[-1]+1:]], dim=1)
    new_A = torch.cat([new_A[:original_node_idx[0]], new_A[original_node_idx[-1]+1:]], dim=0)
    new_A = torch.cat([new_A[:, :original_node_idx[0]], new_A[:, original_node_idx[-1]+1:]], dim=1)

    return new_A, new_H, new_Q