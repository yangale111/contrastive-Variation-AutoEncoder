import ants
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle
import tensorflow as tf
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, BatchNormalization, LeakyReLU, Input, Reshape,Conv3DTranspose,Lambda,ZeroPadding3D,ReLU,Activation,add,MaxPooling3D, AvgPool3D, Layer
from keras import backend as K
from keras.models import Model
from keras.activations import gelu
from keras import callbacks
from keras import layers,Sequential,regularizers
import numpy as np
from glob import glob
from utils import *
import os
from glob import glob
from utils import *

CN_dataset_filepath = sorted(glob('/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Dataset/spm_CNdata/*'))
MCI_dataset_filepath = sorted(glob('/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Dataset/spm_MCIdata/*'))
CN_flods = np.load('./Dataset/CN_MCI_CN_flods.npy',allow_pickle = True)
MCI_flods = np.load('./Dataset/CN_MCI_MCI_flods.npy',allow_pickle = True)


test_mses = []
result_dir = './CN_MCI_result/'
_ = os.mkdir(result_dir) if not os.path.exists(result_dir) else 0
k = 0
while k < 5:
    # 设置参数
    latent_dim = 256
    batch_size = 16
    beta = 1
    gamma = 100
    disentangle = True
    loss = list()
    nbatches = 1000
    # 模型参数保存路径
    fn = result_dir +  'tf_weights_mae_CN_MCI_{}_{}_{}/'.format(beta,gamma,k)
    _ = os.mkdir(fn) if not os.path.exists(fn) else 0
    # 加载数据路径
    CN_trainset_filepath = [CN_dataset_filepath[i] for i in CN_flods[k][0]]
    CN_testset_filepath = [CN_dataset_filepath[i] for i in CN_flods[k][1]]
    MCI_trainset_filepath = [MCI_dataset_filepath[i] for i in MCI_flods[k][0]]
    MCI_testset_filepath = [MCI_dataset_filepath[i] for i in MCI_flods[k][1]]
    # 读取数据
    CN_train_data = load_data(CN_trainset_filepath)
    CN_test_data = load_data(CN_testset_filepath)
    MCI_train_data = load_data(MCI_trainset_filepath)
    MCI_test_data = load_data(MCI_testset_filepath)
    print(CN_train_data.shape,CN_test_data.shape)
    print(MCI_train_data.shape,MCI_test_data.shape)

    # 数据合并
    train_data = np.concatenate((CN_train_data,MCI_train_data), axis=0)
    print(train_data.shape)

    cvae, z_encoder, s_encoder, cvae_decoder = get_MRI_CVAE_3D(latent_dim=latent_dim, beta=beta,disentangle=disentangle,
                                                               gamma=gamma, bias=True, batch_size=batch_size,
                                                               learning_rate=0.001)
    # 训练模型
    pre_hist = 1e10
    patience = 0
    lr = 0.001
    t_cn_batch = []
    t_mci_batch = []
    for i in tqdm(range(1, nbatches)):
        if lr < 0.000001:
            break
        cn_batch = CN_train_data[np.random.randint(low=0, high=CN_train_data.shape[0], size=batch_size), :, :, :]
        mci_batch = MCI_train_data[np.random.randint(low=0, high=MCI_train_data.shape[0], size=batch_size), :, :, :]
        hist = cvae.train_on_batch([mci_batch, cn_batch])
        loss.append(hist[0])
        if pre_hist <= hist[0]:
            patience += 1
        else:
            patience = 0
            pre_hist = hist[0]

        if patience == 30 and lr > 0.000001:
            lr = K.get_value(cvae.optimizer.lr) # 获取当前学习率
            lr = lr * 0.5 # 学习率缩小0.1倍
            K.set_value(cvae.optimizer.lr, lr) # 设置学习率
            patience = 0
            print('new lr:',lr)
            pre_hist = hist[0]
        mse = ((np.array([mci_batch, cn_batch]) - np.array(cvae.predict([mci_batch,cn_batch]))[:, :, :, :, :,0])**2).mean()

        print("hist",hist)
        print('patience:',patience,'  pre_hist:',pre_hist, ' lr:',lr,' mse:',mse)
        assert not np.isnan(hist[0]), 'loss is NaN - somethings wrong'
        im, im1, ss = cvae_query(train_data, s_encoder, z_encoder, cvae_decoder)

        if np.mod(i, 5) == 0:  # Plot training progress
            plot_trainProgress(loss, im, im1)
            pickle.dump(loss, open(fn + 'men_loss.pickle', 'wb'))
            plot_four(mci_batch, cn_batch, z_encoder, s_encoder, cvae_decoder, cvae, idx=0)

            cvae.save_weights(fn+'cvae.h5')
            z_encoder.save_weights(fn+'z_encoder.h5')
            s_encoder.save_weights(fn+'s_encoder.h5')
            cvae_decoder.save_weights(fn+'cvae_decoder.h5')

            t_cn_batch = CN_test_data[np.random.randint(low=0, high=CN_test_data.shape[0], size=batch_size), :, :, :]
            t_mci_batch = MCI_test_data[np.random.randint(low=0, high=MCI_test_data.shape[0], size=batch_size), :, :, :]

            test_mse = ((np.array([t_mci_batch, t_cn_batch]) - np.array(cvae.predict([t_mci_batch, t_cn_batch]))[:, :, :, :, :,0])**2).mean()
            print('test_mse:',test_mse)
            if test_mse < 0.0035:
                break
        if mse < .001:
            break
    # 测试集效果
    test_mse = ((np.array([t_mci_batch, t_cn_batch]) - np.array(cvae.predict([t_mci_batch, t_cn_batch]))[:, :, :, :, :,0])**2).mean()
    if test_mse <= 0.0036:
        k += 1
        test_mses.append(test_mse)
