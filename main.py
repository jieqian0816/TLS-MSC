import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import time
from sklearn import preprocessing
import torch
import slic
import model
from thop import profile
import os
from visual import Draw_Classification_Map,applyPCA,get_Samples_GT,GT_To_One_Hot
import copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
samples_type = ['ratio', 'same_num'][1] 

for (FLAG, curr_train_ratio, Scale, dim, top_k) in [(1,30,60,50,5)]:
# 模型配置IP(1,30,60,50,5), PU(2,30,300,50,5), Sa(3,30,300,50,5), Hou(4,30,600,40,5)
    torch.cuda.empty_cache()
    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    AVG_ALL = []
    F1_ALL =[]
    RECALL_ALL = []
    Train_Time_ALL=[]
    Test_Time_ALL=[]

    Seed_List=[1,3,5,7,9]
    # Seed_List=[41,42,43,44,45,46,47,48,49,50]

    # 数据集读取
    if FLAG == 1:  # IP数据集
        data_mat = sio.loadmat(r'C:\Download\Pycharm\Project\TLS-MSC\datasets\Indian_pines\Indian_pines_corrected.mat')
        data = data_mat['indian_pines_corrected']
        gt_mat = sio.loadmat(r'C:\Download\Pycharm\Project\TLS-MSC\datasets\Indian_pines\Indian_pines_gt.mat')
        gt = gt_mat['indian_pines_gt']

        val_ratio = 0.05  
        class_count = 16  
        learning_rate = 0.0005
        max_epoch = 200
        dataset_name = "indian_"
        pass

    if FLAG == 2:  # PU数据集
        data_mat = sio.loadmat(r'C:\Download\Pycharm\Project\TLS-MSC\datasets\PaviaU\PaviaU.mat')
        data = data_mat['paviaU']
        gt_mat = sio.loadmat(r'C:\Download\Pycharm\Project\TLS-MSC\datasets\PaviaU\PaviaU_gt.mat')
        gt = gt_mat['paviaU_gt']
        
        val_ratio = 0.005  
        class_count = 9  
        learning_rate = 5e-4  
        max_epoch = 100  
        dataset_name = "paviaU_"
        pass

    if FLAG == 3:  # Salinas数据集
        data_mat = sio.loadmat(r'C:\Download\Pycharm\Project\TLS-MSC\datasets\Salinas\Salinas_corrected.mat')
        data = data_mat['salinas_corrected']
        gt_mat = sio.loadmat(r'C:\Download\Pycharm\Project\TLS-MSC\datasets\Salinas\Salinas_gt.mat')
        gt = gt_mat['salinas_gt']

        val_ratio = 0.005
        class_count = 16
        learning_rate = 5e-4
        max_epoch = 100
        dataset_name = "salinas_"
        pass

    if FLAG == 4:  # Houston数据集
        data_mat = sio.loadmat(r'C:\Download\Pycharm\Project\TLS-MSC\datasets\Houston\houston_corrected.mat')
        data = data_mat['houston']
        gt_mat = sio.loadmat(r'C:\Download\Pycharm\Project\TLS-MSC\datasets\Houston\houston_gt.mat')
        train_gt = gt_mat['houston_gt_tr']
        test_gt = gt_mat['houston_gt_te']
        # 合并训练和测试标签矩阵
        merged_gt = test_gt.copy()
        merged_gt[train_gt != 0] = train_gt[train_gt != 0]  # 用 train_gt 的非零值覆盖 test_gt
        gt = merged_gt

        val_ratio = 0.005
        class_count = 15
        learning_rate = 5e-4
        max_epoch = 150
        dataset_name = "houston_"
        pass

    if samples_type == 'same_num': val_ratio = 6 
    superpixel_scale = Scale
    train_ratio = curr_train_ratio
    height, width, bands = data.shape

    # 数据标准化
    orig_data=data
    data = np.reshape(data, [height * width, bands])
    minMax = preprocessing.StandardScaler()  # 对每列数据进行标准化：(x - mean) / std
    data = minMax.fit_transform(data)  # 目的：消除不同波段之间的量纲差异，使数据更适合机器学习模型的训练
    data = np.reshape(data, [height, width, bands])

    gt_reshape=np.reshape(gt, [-1])
    samplesCount_list = []  # 用于存储每个类别的样本数量
    for i in range(class_count):
        idx = np.where(gt_reshape == i + 1)[-1]  # 是一个一维数组，包含当前类别所有样本的索引
        samplesCount = len(idx)  # 统计当前类别的样本数量
        samplesCount_list.append(samplesCount)
    print('输出每个类别的样本数量', samplesCount_list)

    # 创建目录保存结果
    model_dir = './results/' + dataset_name + "/"
    if not os.path.isdir(model_dir):
         os.makedirs(model_dir, exist_ok=True)
    
    Draw_Classification_Map(label=gt, name=model_dir+dataset_name+'gt') # ground truth

    for exp, curr_seed in enumerate(Seed_List):
        print("==============================================================================================")
        print("========================================第", exp+1, "轮实验============================================")
        print("==============================================================================================")
        train_samples_gt, test_samples_gt, val_samples_gt= get_Samples_GT(curr_seed, gt, class_count, curr_train_ratio,val_ratio, samples_type)
        train_samples_gt_onehot = GT_To_One_Hot(train_samples_gt, class_count)
        test_samples_gt_onehot = GT_To_One_Hot(test_samples_gt, class_count)
        val_samples_gt_onehot = GT_To_One_Hot(val_samples_gt, class_count)
        train_samples_gt_onehot = np.reshape(train_samples_gt_onehot, [-1, class_count]).astype(int)
        test_samples_gt_onehot = np.reshape(test_samples_gt_onehot, [-1, class_count]).astype(int)
        val_samples_gt_onehot = np.reshape(val_samples_gt_onehot, [-1, class_count]).astype(int)
        Test_GT = np.reshape(test_samples_gt, [height, width])
        train_val_test_gt=[train_samples_gt,val_samples_gt,test_samples_gt]
        train_label_mask = np.zeros([height * width, class_count])
        temp_ones = np.ones([class_count])
        train_gt_HW = train_samples_gt
        train_samples_gt = np.reshape(train_samples_gt, [height * width])
        for i in range(height * width):
            if train_samples_gt[i] != 0:
                train_label_mask[i] = temp_ones

        train_label_mask = np.reshape(train_label_mask, [height* width, class_count])
        test_label_mask = np.zeros([height * width, class_count])
        temp_ones = np.ones([class_count])
        test_samples_gt = np.reshape(test_samples_gt, [height * width])
        for i in range(height * width):
            if test_samples_gt[i] != 0:
                test_label_mask[i] = temp_ones

        test_label_mask = np.reshape(test_label_mask, [height* width, class_count])
        val_label_mask = np.zeros([height * width, class_count])
        temp_ones = np.ones([class_count])
        val_samples_gt = np.reshape(val_samples_gt, [height * width])
        for i in range(height * width):
            if val_samples_gt[i] != 0:
                val_label_mask[i] = temp_ones

        val_label_mask = np.reshape(val_label_mask, [height* width, class_count])

        # SLIC超像素分割
        ls = slic.LDA_SLIC(data, np.reshape( train_samples_gt,[height,width]), class_count-1)
        tic0=time.time()
        Q, S, A, Seg = ls.simple_superpixel_no_LDA(scale=superpixel_scale)

        masks_all = []
        for node_idx in range(A.shape[0]):
            mk = np.zeros_like(Seg)
            mk[Seg == node_idx] = 1  # 标记当前超像素区域
            masks_all.append(mk)
        # 可视化超像素分割结果
        Draw_Classification_Map(label=gt, name=model_dir + dataset_name+'gt_seg', segments=masks_all)
        toc0 = time.time()
       
        Q=torch.from_numpy(Q).to(device)
        A=torch.from_numpy(A).to(device)
        Seg = torch.from_numpy(Seg).to(device)

        train_samples_gt=torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
        test_samples_gt=torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
        val_samples_gt=torch.from_numpy(val_samples_gt.astype(np.float32)).to(device)

        train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
        test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)
        val_samples_gt_onehot = torch.from_numpy(val_samples_gt_onehot.astype(np.float32)).to(device)

        train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
        test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)
        val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(device)

        # 模型输入
        net_input=np.array(data,np.float32)
        net_input=torch.from_numpy(net_input.astype(np.float32)).to(device)
        net = model.TLS_MSC(height, width, bands, dim, class_count, Q, Seg, train_samples_gt, top_k)
        # 打印参数信息
        print("parameters", net.parameters(), len(list(net.parameters())))
        net.to(device)
        MACs, params = profile(net, inputs=(A, net_input), verbose=False)
        print('MACs: %0.2f M, params: %0.4f K' % (MACs / 1e6, params / 1e3))


        def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor):
            real_labels = reallabel_onehot
            we = -torch.mul(real_labels,torch.log(predict))
            we = torch.mul(we, reallabel_mask) 
            pool_cross_entropy = torch.sum(we)
            return pool_cross_entropy
        

        zeros = torch.zeros([height * width]).to(device).float()
        def evaluate_performance(network_output, train_samples_gt, train_samples_gt_onehot, require_AA_KPP=False,
                                 printFlag=True):
            if False == require_AA_KPP:
                with torch.no_grad():
                    available_label_idx = (train_samples_gt != 0).float()
                    available_label_count = available_label_idx.sum()
                    correct_prediction = torch.where(
                        torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1),
                        available_label_idx, zeros).sum()
                    OA = correct_prediction.cpu() / available_label_count

                    return OA
            else:
                with torch.no_grad():
                    # 原始OA计算保持不变
                    available_label_idx = (train_samples_gt != 0).float()
                    available_label_count = available_label_idx.sum()
                    correct_prediction = torch.where(
                        torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1),
                        available_label_idx, zeros).sum()
                    OA = correct_prediction.cpu() / available_label_count
                    OA = OA.cpu().numpy()

                    # 转换为numpy数组进行处理
                    zero_vector = np.zeros([class_count])
                    output_data = network_output.cpu().numpy()
                    train_samples_gt = train_samples_gt.cpu().numpy()
                    train_samples_gt_onehot = train_samples_gt_onehot.cpu().numpy()
                    output_data = np.reshape(output_data, [height * width, class_count])
                    idx = np.argmax(output_data, axis=-1)

                    for z in range(output_data.shape[0]):
                        if ~(zero_vector == output_data[z]).all():
                            idx[z] += 1

                    # 计算各类别的统计量
                    count_perclass = np.zeros([class_count])
                    correct_perclass = np.zeros([class_count])
                    true_positives = np.zeros([class_count])
                    false_negatives = np.zeros([class_count])
                    false_positives = np.zeros([class_count])

                    for x in range(len(train_samples_gt)):
                        if train_samples_gt[x] != 0:
                            true_class = int(train_samples_gt[x] - 1)
                            pred_class = int(idx[x] - 1)
                            count_perclass[true_class] += 1

                            if train_samples_gt[x] == idx[x]:
                                correct_perclass[true_class] += 1
                                true_positives[true_class] += 1
                            else:
                                false_negatives[true_class] += 1
                                false_positives[pred_class] += 1

                    # 计算各类别精度
                    test_AC_list = correct_perclass / count_perclass
                    test_AA = np.average(test_AC_list)

                    # 计算召回率(Recall)和F1分数
                    recall_perclass = true_positives / (true_positives + false_negatives + 1e-10)
                    precision_perclass = true_positives / (true_positives + false_positives + 1e-10)
                    f1_perclass = 2 * (precision_perclass * recall_perclass) / (
                                precision_perclass + recall_perclass + 1e-10)

                    # 计算宏平均F1和召回率
                    macro_f1 = np.mean(f1_perclass)
                    macro_recall = np.mean(recall_perclass)

                    # Kappa系数计算保持不变
                    test_pre_label_list = []
                    test_real_label_list = []
                    output_data = np.reshape(output_data, [height * width, class_count])
                    idx = np.argmax(output_data, axis=-1)
                    idx = np.reshape(idx, [height, width])

                    for ii in range(height):
                        for jj in range(width):
                            if Test_GT[ii][jj] != 0:
                                test_pre_label_list.append(idx[ii][jj] + 1)
                                test_real_label_list.append(Test_GT[ii][jj])

                    test_pre_label_list = np.array(test_pre_label_list)
                    test_real_label_list = np.array(test_real_label_list)
                    kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16),
                                                      test_real_label_list.astype(np.int16))
                    test_kpp = kappa

                    if printFlag:
                        print("test OA=", OA, "AA=", test_AA, 'kpp=', test_kpp)
                        print("macro F1=", macro_f1, "macro Recall=", macro_recall)
                        print('acc per class:')
                        print(test_AC_list)
                        print('F1 per class:')
                        print(f1_perclass)
                        print('Recall per class:')
                        print(recall_perclass)

                    # 保存所有指标
                    OA_ALL.append(OA)
                    AA_ALL.append(test_AA)
                    KPP_ALL.append(test_kpp)
                    AVG_ALL.append(test_AC_list)
                    F1_ALL.append(macro_f1)
                    RECALL_ALL.append(macro_recall)

                    # 更新结果文件写入内容
                    f = open(model_dir + dataset_name + '_results.txt', 'a+')
                    str_results = '\n===============================' \
                                  + " learning rate=" + str(learning_rate) \
                                  + " epochs=" + str(max_epoch) \
                                  + " train ratio=" + str(train_ratio) \
                                  + " val ratio=" + str(val_ratio) \
                                  + " ===============================" \
                                  + "\nOA=" + str(OA) \
                                  + "\nAA=" + str(test_AA) \
                                  + '\nkpp=' + str(test_kpp) \
                                  + '\nmacro F1=' + str(macro_f1) \
                                  + '\nmacro Recall=' + str(macro_recall) \
                                  + '\nacc per class: ' + str(test_AC_list) \
                                  + '\nF1 per class: ' + str(f1_perclass) \
                                  + '\nRecall per class: ' + str(recall_perclass) \
                                  + '\n'

                    f.write(str_results)
                    f.close()
                    return OA

        optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50) # 50个epoch内按余弦曲线调整

        best_loss=99999
        best_OA=0
        Stop_Flag=0
        net.train()
        tic1 = time.perf_counter()
        
        for i in range(max_epoch+1):
            optimizer.zero_grad()
            output, *_ = net(A, net_input)
            loss = compute_loss(output,train_samples_gt_onehot,train_label_mask)
            torch.autograd.set_detect_anomaly(True)  # 在backward前添加
            loss.backward(retain_graph=False)
            
            optimizer.step()  
            if i%10==0:
                with torch.no_grad():
                    net.eval()
                    output, *_ = net(A, net_input)
                    trainloss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
                    trainOA = evaluate_performance(output, train_samples_gt, train_samples_gt_onehot)
                    valloss = compute_loss(output, val_samples_gt_onehot, val_label_mask)
                    valOA = evaluate_performance(output, val_samples_gt, val_samples_gt_onehot)
                    print("{}\ttrain loss={}\t train OA={} val loss={}\t val OA={}".format(str(i + 1), trainloss, trainOA, valloss, valOA))

                    if valloss < best_loss  or valOA>best_OA:
                        best_loss = valloss
                        best_OA=valOA
                        Stop_Flag=0
                        BestNet = copy.deepcopy(net).to("cpu")
                        
                        print('save model...')
                torch.cuda.empty_cache()
                net.train()
        toc1 = time.perf_counter()

        print("\n=========================training done. starting evaluation...================================")

        training_time=toc1 - tic1  
        Train_Time_ALL.append(training_time)
        
        torch.cuda.empty_cache()
        with (torch.no_grad()):
            
            net = BestNet.to(device)
            net.eval()
            tic2 = time.perf_counter()
            output, out_feat = net(A, net_input)
            toc2 = time.perf_counter()
            testloss = compute_loss(output, test_samples_gt_onehot, test_label_mask)
            testOA = evaluate_performance(output, test_samples_gt, test_samples_gt_onehot,require_AA_KPP=True,printFlag=False)
            print("{}\ttest loss={}\t test OA={}".format(str(i + 1), testloss, testOA))
            classification_map = torch.argmax(output, 1).reshape([height,width]).cpu()+1

            Draw_Classification_Map(label=classification_map, name=model_dir + dataset_name + 'background' + str(testOA))

            ignored_labels = [0]
            mask = np.zeros(gt.shape, dtype='bool')
            for l in ignored_labels:
                mask[gt == l] = True
            classification_map[mask] = 0  # 忽略背景
            Draw_Classification_Map(label=classification_map, name=model_dir + dataset_name + str(testOA))

            testing_time=toc2 - tic2
            Test_Time_ALL.append(testing_time)

            ################### t-SNE可视化 ####################
            labels = test_samples_gt.reshape(-1).cpu().numpy()  # 展平为[H*W]
            features = out_feat.cpu().numpy()  # 形状: [H*W, C]
            valid_mask = labels > 0  # 过滤无效标签（如背景类标签为0或-1）
            features = features[valid_mask]
            labels = labels[valid_mask]
            unique_labels, counts = np.unique(labels, return_counts=True)
            sampled_indices = []
            for class_id, count in zip(unique_labels, counts):
                class_indices = np.where(labels == class_id)[0]
                n_samples = min(500, count)  # 每类最多采样500个点, 分层采样（公平比较）
                sampled_indices.extend(np.random.choice(class_indices, n_samples, replace=False))

            features_sampled = features[sampled_indices]
            labels_sampled = labels[sampled_indices]
            features_scaled = StandardScaler().fit_transform(features_sampled) # 标准化特征
            tsne = TSNE(
                n_components=2,
                perplexity=30,
                learning_rate=200,
                random_state=42
            )
            features_tsne = tsne.fit_transform(features_scaled)
            plt.figure(figsize=(12, 10))
            # 可视化, 为少数类分配更大的点和更鲜艳的颜色（动态调整点大小和颜色）
            cmap = plt.get_cmap('tab20', len(unique_labels))  # viridis
            for i, class_id in enumerate(unique_labels):
                mask = (labels_sampled == class_id)
                size = 100 if counts[i] < 100 else 50  # 少数类点更大
                plt.scatter(
                    features_tsne[mask, 0],
                    features_tsne[mask, 1],
                    c=[cmap(i)],
                    label=f'Class {class_id} (n={counts[i]})',
                    s=size,
                    alpha=0.7,
                    edgecolors='w',
                    linewidths=0.5
                )
            plt.axis('off')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='12', shadow=True, borderpad=1)
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.savefig(
                f"{model_dir}{dataset_name}_T-SNE_{testOA}.png",dpi=300,
                bbox_inches='tight',
                transparent=True
            )
            plt.close()  # 避免内存泄漏

        torch.cuda.empty_cache()
        del net

    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    AVG_ALL = np.array(AVG_ALL)
    F1_ALL = np.array(F1_ALL)
    RECALL_ALL = np.array(RECALL_ALL)
    Train_Time_ALL=np.array(Train_Time_ALL)
    Test_Time_ALL=np.array(Test_Time_ALL)


    print("\ntrain_ratio={}".format(curr_train_ratio), "\n=====================================================================================")
    print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))
    print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
    print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
    print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
    print('F1=', np.mean(F1_ALL), '+-', np.std(F1_ALL))
    print('RECALL=', np.mean(RECALL_ALL), '+-', np.std(RECALL_ALL))
    print("Average training time:{}".format(np.mean(Train_Time_ALL)))
    print("Average testing time:{}".format(np.mean(Test_Time_ALL)))
    
    # 保存数据信息
    f = open(model_dir + dataset_name + '_results.txt', 'a+')
    str_results = '\n***************************************************************' \
    +"\ntrain_ratio={}".format(curr_train_ratio) \
    +'\nOA='+ str(np.mean(OA_ALL))+ '+-'+ str(np.std(OA_ALL)) \
    +'\nAA='+ str(np.mean(AA_ALL))+ '+-'+ str(np.std(AA_ALL)) \
    +'\nKpp='+ str(np.mean(KPP_ALL))+ '+-'+ str(np.std(KPP_ALL)) \
    +'\nAVG='+ str(np.mean(AVG_ALL,0))+ '+-'+ str(np.std(AVG_ALL,0)) \
    +'\nF1='+ str(np.mean(F1_ALL))+ '+-'+ str(np.std(F1_ALL)) \
    +'\nRECALL='+ str(np.mean(RECALL_ALL))+ '+-'+ str(np.std(RECALL_ALL)) \
    +"\nAverage training time:{}".format(np.mean(Train_Time_ALL)) \
    +"\nAverage testing time:{}".format(np.mean(Test_Time_ALL)) 
    f.write(str_results)
    f.close()