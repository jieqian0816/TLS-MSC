"""SLIC regions and the fixed Gaussian graph in manuscript Eqs. (7)--(8)."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from skimage.segmentation import slic


def SegmentsLabelProcess(labels):
    """Reindex arbitrary region IDs to contiguous zero-based IDs."""
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError('segments must be a two-dimensional image')
    return np.unique(labels, return_inverse=True)[1].reshape(labels.shape)


def spatial_neighbor_pairs(segments):
    """All undirected shared four-connected boundaries, including image borders."""
    segments = np.asarray(segments)
    if segments.ndim != 2:
        raise ValueError('segments must be a two-dimensional image')
    horizontal = np.column_stack((segments[:, :-1].ravel(), segments[:, 1:].ravel()))
    vertical = np.column_stack((segments[:-1, :].ravel(), segments[1:, :].ravel()))
    pairs = np.concatenate((horizontal, vertical), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return np.unique(np.sort(pairs, axis=1), axis=0).astype(np.int64)


def gaussian_affinity_graph(segments, node_spectra, gamma=1.0):
    """A_ij = exp(-gamma * mean_b((p_ib - p_jb)**2)); A_ii = 0.

    node_spectra are region means of standardized input spectra. Dividing
    squared Euclidean distances by B is exactly the manuscript's x_i=p_i/sqrt(B).
    Region IDs must index rows of node_spectra. This function takes no labels.
    """
    segments = np.asarray(segments)
    spectra = np.asarray(node_spectra, dtype=np.float64)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError('gamma must be finite and strictly positive')
    if spectra.ndim != 2 or spectra.shape[1] == 0 or not np.isfinite(spectra).all():
        raise ValueError('node_spectra must be a finite M-by-B matrix')
    if not np.array_equal(np.unique(segments), np.arange(len(spectra))):
        raise ValueError('segments must use contiguous IDs 0,...,M-1')
    adjacency = np.zeros((len(spectra), len(spectra)), dtype=np.float32)
    pairs = spatial_neighbor_pairs(segments)
    if len(pairs):
        delta = spectra[pairs[:, 0]] - spectra[pairs[:, 1]]
        weights = np.exp(-float(gamma) * np.mean(delta * delta, axis=1)).astype(np.float32)
        if not np.all(weights > 0):
            raise ValueError('Gaussian weights underflowed; choose a smaller gamma')
        adjacency[pairs[:, 0], pairs[:, 1]] = weights
        adjacency[pairs[:, 1], pairs[:, 0]] = weights
    return adjacency


class SLIC:
    def __init__(self, HSI, labels=None, n_segments=1000, compactness=20,
                 max_iter=20, sigma=0, min_size_factor=0.3, max_size_factor=2):
        image = np.asarray(HSI)
        if image.ndim != 3 or not np.isfinite(image).all():
            raise ValueError('HSI must be a finite H-by-W-by-B image')
        h, w, bands = image.shape
        self.data = StandardScaler().fit_transform(image.reshape(-1, bands)).reshape(h, w, bands)
        self.n_segments = min(h * w, max(1, int(round(n_segments))))
        self.compactness = compactness
        self.max_iter = max_iter
        self.sigma = sigma  # Spatial smoothing for SLIC, NOT graph gamma.
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor
        # labels is accepted for source compatibility but is never read.

    def get_Q_and_S_and_Segments(self):
        segments = slic(self.data, n_segments=self.n_segments,
                        compactness=self.compactness, max_num_iter=self.max_iter,
                        convert2lab=False, sigma=self.sigma,
                        enforce_connectivity=True, min_size_factor=self.min_size_factor,
                        max_size_factor=self.max_size_factor, slic_zero=False,
                        start_label=0, channel_axis=-1)
        self.segments = SegmentsLabelProcess(segments)
        ids = self.segments.ravel()
        self.superpixel_count = int(ids.max()) + 1
        counts = np.bincount(ids, minlength=self.superpixel_count)
        sums = np.zeros((self.superpixel_count, self.data.shape[-1]), dtype=np.float64)
        np.add.at(sums, ids, self.data.reshape(-1, self.data.shape[-1]))
        self.S = (sums / counts[:, None]).astype(np.float32)
        self.Q = np.zeros((ids.size, self.superpixel_count), dtype=np.float32)
        self.Q[np.arange(ids.size), ids] = 1.0
        return self.Q, self.S, self.segments

    def get_A(self, gamma=1.0):
        return gaussian_affinity_graph(self.segments, self.S, gamma)


class LDA_SLIC:
    """Legacy wrapper. The paper path is simple_superpixel_no_LDA only."""
    def __init__(self, data, labels=None, n_component=None):
        self.data = np.asarray(data)
        self.init_labels = labels
        self.n_component = n_component
        self.height, self.width, self.bands = self.data.shape

    def SLIC_Process(self, img, scale=25, gamma=1.0):
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('segmentation scale must be positive')
        segmenter = SLIC(img, n_segments=self.height * self.width / scale,
                         compactness=1, sigma=1, min_size_factor=0.1, max_size_factor=2)
        Q, S, segments = segmenter.get_Q_and_S_and_Segments()
        return Q, S, segmenter.get_A(gamma), segments

    def simple_superpixel_no_LDA(self, scale, gamma=1.0):
        return self.SLIC_Process(self.data, scale=scale, gamma=gamma)

    def LDA_Process(self, curr_labels):
        """Optional legacy transform; not used by the manuscript experiments."""
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        labels = np.asarray(curr_labels).ravel()
        valid = labels > 0
        data = self.data.reshape(-1, self.bands)
        transformed = LinearDiscriminantAnalysis(n_components=self.n_component).fit(
            data[valid], labels[valid]).transform(data)
        return transformed.reshape(self.height, self.width, -1)

    def simple_superpixel(self, scale, gamma=1.0):
        return self.SLIC_Process(self.LDA_Process(self.init_labels), scale, gamma)
