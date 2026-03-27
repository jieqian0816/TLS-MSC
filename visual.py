import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import random
import spectral as spy
from sklearn.decomposition import PCA


def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX


def get_Samples_GT(seed: int, gt: np.array, class_count: int, train_ratio,val_ratio, samples_type: str = 'ratio', ):

    random.seed(seed)
    [height, width] = gt.shape
    gt_reshape = np.reshape(gt, [-1]) 
    train_rand_idx = []
    val_rand_idx = []
    per_class_train = []
    if samples_type == 'ratio':
        for i in range(class_count):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            rand_list = [i for i in range(samplesCount)]  
            rand_idx = random.sample(rand_list,
                                        np.ceil(samplesCount * train_ratio).astype('int32'))  
            rand_real_idx_per_class = idx[rand_idx]
            per_class_train.append(len(rand_real_idx_per_class))
            train_rand_idx.append(rand_real_idx_per_class)
        train_rand_idx = np.array(train_rand_idx)
        train_data_index = []
        
        for c in range(train_rand_idx.shape[0]):
            a = train_rand_idx[c]
            for j in range(a.shape[0]):
                train_data_index.append(a[j])
        train_data_index = np.array(train_data_index)
        
        
        train_data_index = set(train_data_index)
        all_data_index = [i for i in range(len(gt_reshape))]
        all_data_index = set(all_data_index)
        
       
        background_idx = np.where(gt_reshape == 0)[-1]
        background_idx = set(background_idx)
        test_data_index = all_data_index - train_data_index - background_idx
        
        
        val_data_count = int(val_ratio * (len(test_data_index) + len(train_data_index)))  
        val_data_index = random.sample(test_data_index, val_data_count)
        val_data_index = set(val_data_index)
        test_data_index = test_data_index - val_data_index  
        
        
        test_data_index = list(test_data_index)
        train_data_index = list(train_data_index)
        val_data_index = list(val_data_index)
        
    if samples_type == 'same_num':
        for i in range(class_count):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            real_train_samples_per_class = int(train_ratio)  
            rand_list = [i for i in range(samplesCount)]  
            if real_train_samples_per_class > samplesCount:  
                real_train_samples_per_class = samplesCount
            rand_idx = random.sample(rand_list,
                                        real_train_samples_per_class)  
            
            h = np.sum(gt==(i+1))
            
            if real_train_samples_per_class >= h:
                real_train_samples_per_class=15
                
            rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
            train_rand_idx.append(rand_real_idx_per_class_train)

        train_rand_idx = np.array(train_rand_idx, dtype=object)
        val_rand_idx = np.array(val_rand_idx)
        train_data_index = []
        for c in range(train_rand_idx.shape[0]):
            a = train_rand_idx[c]
            for j in range(a.shape[0]):
                train_data_index.append(a[j])
        train_data_index = np.array(train_data_index)
        train_data_index = set(train_data_index)
        all_data_index = [i for i in range(len(gt_reshape))]
        all_data_index = set(all_data_index)
        background_idx = np.where(gt_reshape == 0)[-1]
        background_idx = set(background_idx)
        test_data_index = all_data_index - train_data_index - background_idx
        val_data_count = int(class_count)
        val_data_index = random.sample(list(test_data_index), val_data_count)
        val_data_index = set(val_data_index)
        test_data_index = list(test_data_index)
        train_data_index = list(train_data_index)
        val_data_index = list(val_data_index)

    train_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(train_data_index)):
        train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
        pass

    test_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(test_data_index)):
        test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
        pass
    
    Test_GT = np.reshape(test_samples_gt, [height, width])  

    val_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(val_data_index)):
        val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
        pass
    
    test_per_class = []
    val_per_class=[]
    train_per_class=[]
    for i in range(class_count):
        test_per_class.append(sum(test_samples_gt==(i+1)))
        val_per_class.append(sum(val_samples_gt==(i+1)))
        train_per_class.append(sum(train_samples_gt==(i+1)))

    train_samples_gt=np.reshape(train_samples_gt,[height, width])
    test_samples_gt=np.reshape(test_samples_gt,[height, width])
    val_samples_gt=np.reshape(val_samples_gt,[height, width])

    return train_samples_gt, test_samples_gt, val_samples_gt


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


def Draw_Classification_Map(label, name: str, scale: float = 4.0, dpi: int = 400, segments=None):

        fig, ax = plt.subplots()
        numlabel = np.array(label)
        # 如果需要对图像进行进一步操作（例如设置标题、调整颜色条等），可以使用 v
        v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number, cmap='jet')

        # 超像素分割叠加边界
        if segments is not None:
            for mk in segments:
                plt.contour(mk, colors='white', linewidths=0.15)

        ax.set_axis_off()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
        foo_fig = plt.gcf()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
        plt.close(fig)
        pass
