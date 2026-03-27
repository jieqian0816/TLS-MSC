import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN, self).__init__()
        p = 0.3
        self.proj = nn.Linear(in_dim, out_dim)
        self.BN = nn.BatchNorm1d(in_dim)
        self.act = nn.LeakyReLU(inplace=True)
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

    def A_to_D_inv(self, g):
        D = g.sum(1)  # 按行求和，计算每个节点的度数
        D_hat = torch.diag(torch.pow(D, -0.5))  # D先开根号，再扩为对角矩阵
        return D_hat

    def forward(self, g, h):
        h = self.BN(h)
        # 邻接矩阵 + 单位矩阵
        g = g + torch.eye(g.shape[0], g.shape[0], requires_grad=False, device=device, dtype=torch.float32)
        D_hat = self.A_to_D_inv(g)
        h = torch.matmul(torch.matmul(torch.matmul(D_hat, g), D_hat), h)
        h = self.proj(h)  # 这里投影层 h = h * W + b
        h = self.act(h)
        return h


class Topo_Label_Score(nn.Module):
    def __init__(self, k, in_dim, node_labels):
        super(Topo_Label_Score, self).__init__()
        self.k = k  # top_k节点，要使用MSC的节点数量
        self.node_labels = node_labels
        self.sigmoid = nn.Sigmoid()
        self.BN = nn.BatchNorm1d(in_dim)
        self.softmax = nn.Softmax()
        self.proj = nn.Linear(in_dim, 1)
        # 定义一个可学习的参数 sigma1，初始值为 0.5，在训练过程中会自动更新。
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True))

    def label_scores(
            self,
            adj_matrix: torch.Tensor,  # [N, N] 邻接矩阵 (0/1或权重)
            node_labels: torch.Tensor,  # [N] 节点类别标签 (0~C-1)
            superpixel_features: torch.Tensor,  # [N, D] 超像素的光谱特征 (N个超像素，D个波段)
            alpha: float = 0.5,  # 标签评分和熵评分权重 (0~1)
            use_entropy: bool = False,  # 是否使用熵 (默认用方差)
            bins: int = 10  # 熵计算的分箱数
    ) -> torch.Tensor:
        """
        计算标签节点评分:
            1. 基于邻居类别统计标记边界节点
            2. 计算光谱异质性 (方差或熵)
            3. 标签评分 = alpha * 边界得分 + (1-alpha) * 异质性得分
        Returns:
            scores (torch.Tensor): [N], 评分越高越可能是混合边界节点
        """
        device = adj_matrix.device
        N = adj_matrix.shape[0]

        # --- 1. 边界节点检测 ---
        boundary_mask = torch.zeros(N, dtype=torch.bool, device=device)

        for i in range(N):
            neighbors = adj_matrix[i].nonzero().squeeze(-1)  # 邻居索引
            neighbor_labels = node_labels[neighbors]
            unique_labels = torch.unique(neighbor_labels)
            boundary_mask[i] = len(unique_labels) > 1  # 多类别邻居 -> 边界

        boundary_scores = boundary_mask.float()  # 边界得分 (0或1)

        # --- 2. 异质性计算 (方差或熵) ---
        if use_entropy:
            # 计算熵 (需转到CPU用scipy)
            features_np = superpixel_features.cpu().detach().numpy()
            heterogeneity = torch.zeros(N, device=device)
            for i in range(N):
                hist = np.histogram(features_np[i], bins=bins)[0]
                hist = hist / hist.sum() + 1e-10  # 避免除零
                heterogeneity[i] = torch.tensor(entropy(hist), device=device)
        else:
            # 计算方差 (PyTorch直接支持)
            heterogeneity = torch.var(superpixel_features, dim=1)

        # 归一化异质性到 [0, 1]
        heterogeneity = (heterogeneity - heterogeneity.min()) / \
                        (heterogeneity.max() - heterogeneity.min() + 1e-10)

        # --- 3. 标签评分 ---
        scores = alpha * boundary_scores + (1 - alpha) * heterogeneity

        return scores

    def top_k(self, scores, k):
        topk_values, topk_indices = torch.topk(scores, k=k)
        return topk_indices

    def forward(self, g, h):
        h = self.BN(h)
        D = g.sum(1)  # 计算邻接矩阵中节点的权重和
        D_hat = torch.diag(torch.pow(D, -1))
        Z1 = torch.abs(h - torch.matmul(torch.matmul(D_hat, g), h)).sum(dim=1)
        Z2 = self.sigmoid(D)
        pl = Z1 + Z2    # 局部拓扑评分

        Z3 = torch.matmul(torch.matmul(D_hat, g), h)
        Z3 = self.proj(Z3).squeeze()
        pg = F.softmax(Z3, dim=-1)  # 全局拓扑评分
        pt = pl + pg   # 拓扑评分
        # 标签评分
        p_label = self.label_scores(g, self.node_labels, h, alpha=0.6, use_entropy=True)
        total_score = torch.sigmoid(pt * p_label)

        # topk
        topk_indices = self.top_k(total_score, self.k)

        return g, h, topk_indices


def superpixel_labels(train_samples_gt: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
    """
    为每个超像素生成伪标签（当且仅当超像素内所有有标签像素属于同一类别时）

    参数:
        train_samples_gt: 形状 [H*W] 的标签张量 (0-15为有效类别，0表示无标签)
        segments: 形状 [H, W] 的超像素分割矩阵，值代表超像素ID

    返回:
        pseudo_labels: 形状 [num_superpixels] 的伪标签张量 (0表示无标签)
    """
    # 获取超像素ID列表
    superpixel_ids = torch.unique(segments)
    num_superpixels = len(superpixel_ids)

    # 初始化伪标签张量
    pseudo_labels = torch.zeros(num_superpixels, dtype=torch.long)

    # 将segments展平以匹配train_samples_gt
    segments_flat = segments.flatten()  # [H*W]

    for sp_id in superpixel_ids:
        # 获取当前超像素的所有像素索引
        pixel_indices = (segments_flat == sp_id).nonzero().squeeze(-1)

        # 提取这些像素的标签（过滤掉0标签）
        labels = train_samples_gt[pixel_indices]
        valid_labels = labels[labels != 0]

        # 如果当前超像素没有有效标签，跳过
        if len(valid_labels) == 0:
            continue

        # 检查所有有效标签是否相同
        unique_labels = torch.unique(valid_labels)
        if len(unique_labels) == 1:
            pseudo_labels[sp_id] = unique_labels.item()
        # 否则保持为0（无标签）

    pseudo_labels = pseudo_labels.to(device)
    return pseudo_labels


def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g


def extract_superpixel_regions(data, superpixel_indices, mask_matrix):
    region_list = []
    region_info_list = []
    for index in superpixel_indices:
        # 找出当前超像素节点所包含的所有原始像素的坐标
        superpixel_coords = torch.nonzero(mask_matrix == index, as_tuple=False)
        if superpixel_coords.numel() == 0:
            continue

        # 找出边界坐标
        min_row = superpixel_coords[:, 0].min()
        max_row = superpixel_coords[:, 0].max()
        min_col = superpixel_coords[:, 1].min()
        max_col = superpixel_coords[:, 1].max()
        # 提取包含当前超像素的最小矩形区域
        region = data[min_row:max_row + 1, min_col:max_col + 1, :]
        region_list.append(region)
        # 获取区域内所有原始像素的坐标(相对于原图的绝对坐标)
        original_coords = [(int(coord[0]), int(coord[1])) for coord in superpixel_coords]
        region_info_list.append((region, original_coords))  # 保存区域数据和坐标信息
    return region_list, region_info_list


def write_back_features(original_data, results, processed_features):
    """
    将处理后的特征根据矩形区域内的坐标位置回写到原始数据中
    参数:
        original_data: 原始数据张量 (H, W, C) [会被in-place修改]
        results: extract_superpixel_regions()的输出结果，每个元素包含：
            - 矩形区域数据 (h, w, C)
            - 该区域内所有原始像素的坐标列表 [(row1, col1), (row2, col2), ...]
        processed_features: 处理后的特征列表，每个元素对应一个区域的调整后特征 (h, w, C)
    返回:
        修改后的原始数据 (H, W, C)
    """
    modified_data = original_data.clone()
    for (region, coords), features in zip(results, processed_features):
        # 获取当前区域的左上角坐标
        min_row = min(r for (r, _) in coords)
        min_col = min(c for (_, c) in coords)
        # 遍历每个坐标并回写特征
        for (abs_r, abs_c) in coords:
            # 计算相对坐标
            rel_r = abs_r - min_row
            rel_c = abs_c - min_col
            # 从处理后的特征中提取对应位置的特征
            modified_data[abs_r, abs_c] = features[rel_r, rel_c, :]

    return modified_data


class Mixed_Superpixel_CNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.BN = nn.BatchNorm1d(in_dim)
        mid_dim = 64
        # CNN特征提取器
        self.cnn = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_dim, mid_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(mid_dim),
            nn.Conv2d(mid_dim, out_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Identity()  # 保持维度
        )
        # 定义一个可学习的参数 sigma1，初始值为 0.5，在训练过程中会自动更新。
        self.sigma1= torch.nn.Parameter(torch.tensor([0.5],requires_grad=True))

    def forward(self, data, boundary_nodes, segments_mask):
        """
        Args:
            data: 原始高光谱图像 (H, W, B)
            boundary_nodes: 边界节点索引列表 [n]
            h: GCN提取的特征 (N, d)
            segments_mask: 超像素标签图 (H, W)
        Returns:
            updated_features: 更新后的边界节点特征 (n, d)
        """
        region_list, region_info_list = extract_superpixel_regions(data, boundary_nodes, segments_mask)
        patch_features = []
        for region_idx in range(len(region_list)):
            # 1. 提取当前节点超像素区域数据
            patch = region_list[region_idx]
            patch = patch.to(device).unsqueeze(0)
            patch = patch.permute(0, 3, 1, 2)
            # 2. CNN特征提取器
            cnn_feat = self.cnn(patch)
            patch = cnn_feat.permute(0, 2, 3, 1)
            patch = patch.squeeze(0)
            patch_features.append(torch.sigmoid(patch))

        return patch_features, region_info_list


class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(SSConv, self).__init__()
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch // 2,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.Act1 = nn.LeakyReLU(inplace=True)
        self.Act2 = nn.LeakyReLU(inplace=True)
        self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, input):
        out = self.BN(input)
        out = self.point_conv(out)
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out


class TLS_MSC(nn.Module):
    def __init__(self, height, width, in_dim, dim, n_classes, Q, segments, train_gt, top_k):
        super().__init__()
        self.height = height
        self.width = width
        self.in_dim = in_dim
        self.n_classes = n_classes  # 类别数
        self.Q = Q
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化Q
        self.segments = segments
        self.train_gt = train_gt
        self.node_labels = superpixel_labels(self.train_gt, self.segments)
        self.sigma_1 = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.sigma_2 = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True))

        # 隐藏层维数
        mid_channel = dim

        # CNN去噪
        self.CNN_denoise = nn.Sequential()
        self.CNN_denoise.add_module('CNN_denoise_BN1', nn.BatchNorm2d(self.in_dim))
        self.CNN_denoise.add_module('CNN_denoise_Conv1',nn.Conv2d(self.in_dim, mid_channel, kernel_size=(1, 1)))
        self.CNN_denoise.add_module('CNN_denoise_Act1', nn.LeakyReLU())
        self.CNN_denoise.add_module('CNN_denoise_BN2', nn.BatchNorm2d(mid_channel))
        self.CNN_denoise.add_module('CNN_denoise_Conv2', nn.Conv2d(mid_channel, mid_channel, kernel_size=(1, 1)))
        self.CNN_denoise.add_module('CNN_denoise_Act2', nn.LeakyReLU())

        # GCN
        self.gcn1 = GCN(mid_channel, mid_channel)
        self.gcn2 = GCN(mid_channel, mid_channel)
        self.gcn3 = GCN(mid_channel, mid_channel)

        # 拓扑标签评分（TLS）
        self.score1 = Topo_Label_Score(top_k, mid_channel, self.node_labels)
        self.score2 = Topo_Label_Score(top_k, mid_channel, self.node_labels)

        # 节点CNN（MSC）
        self.NodeCNN1 = Mixed_Superpixel_CNN(mid_channel, mid_channel)
        self.NodeCNN2 = Mixed_Superpixel_CNN(mid_channel, mid_channel)

        # 全局空间光谱CNN
        self.conv1 = SSConv(mid_channel, mid_channel, kernel_size=5)
        self.conv2 = SSConv(mid_channel, mid_channel, kernel_size=5)
        self.conv3 = SSConv(mid_channel, mid_channel, kernel_size=5)

        # 分类头
        self.classifier = nn.Linear(mid_channel, self.n_classes)

    def forward(self, g, x):
        (height, width, band) = x.shape
        # CNN去噪和光谱变换
        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        noise = torch.squeeze(noise, 0).permute([1, 2, 0])
        clean_x = noise
        clean_x_flatten = clean_x.reshape([height * width, -1])
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), clean_x_flatten)  # 编码
        org_h = superpixels_flatten

        gcn_h1 = self.gcn1(g, org_h)  # GCN1

        pool_g1, pool_h1, pool_top_idx1 = self.score1(g, gcn_h1)  # scoring1
        patch_features1, region_info_list1 = self.NodeCNN1(clean_x, pool_top_idx1, self.segments)  # MSC1
        gcn_h2 = self.gcn2(g, pool_h1)  # GCN2

        pool_g2, pool_h2, pool_top_idx2 = self.score2(g, gcn_h2)  # scoring2
        patch_features2, region_info_list2 = self.NodeCNN2(clean_x, pool_top_idx2, self.segments)  # MSC2
        gcn_h3 = self.gcn3(g, pool_h2)  # GCN3

        pixels_feat1 = torch.matmul(self.Q, gcn_h2)  # 解码1
        pixels_feat1 = pixels_feat1.reshape([height, width, -1])
        pixels_feat1 = torch.sigmoid(pixels_feat1)

        # 将边界超像素节点处理后的特征写回原图
        modified_data1 = write_back_features(pixels_feat1, region_info_list1, patch_features1)

        pixels_feat2 = torch.matmul(self.Q, gcn_h3)  # 解码2
        pixels_feat2 = pixels_feat2.reshape([height, width, -1])
        pixels_feat2 = torch.sigmoid(pixels_feat2)

        # 将边界超像素节点处理后的特征写回原图
        modified_data2 = write_back_features(pixels_feat2, region_info_list2, patch_features2)

        modified_data = clean_x + modified_data1 + modified_data2  # 残差连接

        CNN_input = torch.unsqueeze(modified_data.permute([2, 0, 1]), 0)
        cnn1 = self.conv1(CNN_input)
        cnn_out = torch.squeeze(cnn1, 0).permute([1, 2, 0]).reshape([height * width, -1])

        out = self.classifier(cnn_out)  # 线性降维
        y = F.softmax(out, -1)

        return y, cnn_out
