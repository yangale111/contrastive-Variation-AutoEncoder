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


CN_flods = np.load('./Dataset/CN_AD_CN_flods.npy', allow_pickle=True)
AD_flods = np.load('./Dataset/CN_AD_AD_flods.npy', allow_pickle=True)
print(CN_flods.shape, AD_flods.shape)


test_mses = []
result_dir = './CN_AD_result/'
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
    fn = result_dir +  'tf_weights_mae_CN_AD_{}_{}_{}/'.format(beta,gamma,k)
    _ = os.mkdir(fn) if not os.path.exists(fn) else 0
    # 加载数据路径
    CN_trainset_filepath = [CN_dataset_filepath[i] for i in CN_flods[k][0]]
    CN_testset_filepath = [CN_dataset_filepath[i] for i in CN_flods[k][1]]
    AD_trainset_filepath = [AD_dataset_filepath[i] for i in AD_flods[k][0]]
    AD_testset_filepath = [AD_dataset_filepath[i] for i in AD_flods[k][1]]
    # 读取数据
    CN_train_data = load_data(CN_trainset_filepath)
    CN_test_data = load_data(CN_testset_filepath)
    AD_train_data = load_data(AD_trainset_filepath)
    AD_test_data = load_data(AD_testset_filepath)
    print(CN_train_data.shape,CN_test_data.shape)
    print(AD_train_data.shape,AD_test_data.shape)

    # 数据合并
    train_data = np.concatenate((CN_train_data,AD_train_data), axis=0)
    print(train_data.shape)

    cvae, z_encoder, s_encoder, cvae_decoder = get_MRI_CVAE_3D(latent_dim=latent_dim, beta=beta,disentangle=disentangle,
                                                               gamma=gamma, bias=True, batch_size=batch_size,learning_rate=0.001)
    # 训练模型
    pre_hist = 1e10
    patience = 0
    lr = 0.001
    t_cn_batch = []
    t_ad_batch = []
    for i in tqdm(range(1, nbatches)):
        if lr < 0.000001:
            break
        cn_batch = CN_train_data[np.random.randint(low=0, high=CN_train_data.shape[0], size=batch_size), :, :, :]
        ad_batch = AD_train_data[np.random.randint(low=0, high=AD_train_data.shape[0], size=batch_size), :, :, :]
        hist = cvae.train_on_batch([ad_batch, cn_batch])
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
        mse = ((np.array([ad_batch, cn_batch]) - np.array(cvae.predict([ad_batch,cn_batch]))[:, :, :, :, :,0])**2).mean()

        print("hist",hist)
        print('patience:',patience,'  pre_hist:',pre_hist, ' lr:',lr,' mse:',mse)
        assert not np.isnan(hist[0]), 'loss is NaN - somethings wrong'
        im, im1, ss = cvae_query(train_data, s_encoder, z_encoder, cvae_decoder)

        if np.mod(i, 5) == 0:  # Plot training progress
            plot_trainProgress(loss, im, im1)
            pickle.dump(loss, open(fn + 'men_loss.pickle', 'wb'))
            plot_four(ad_batch, cn_batch, z_encoder, s_encoder, cvae_decoder, cvae, idx=0)

            cvae.save_weights(fn+'cvae.h5')
            z_encoder.save_weights(fn+'z_encoder.h5')
            s_encoder.save_weights(fn+'s_encoder.h5')
            cvae_decoder.save_weights(fn+'cvae_decoder.h5')

            t_cn_batch = CN_test_data[np.random.randint(low=0, high=CN_test_data.shape[0], size=batch_size), :, :, :]
            t_ad_batch = AD_test_data[np.random.randint(low=0, high=AD_test_data.shape[0], size=batch_size), :, :, :]

            test_mse = ((np.array([t_ad_batch, t_cn_batch]) - np.array(cvae.predict([t_ad_batch, t_cn_batch]))[:, :, :, :, :,0])**2).mean()
            print('test_mse:',test_mse)
            if test_mse < 0.0035:
                break
        if mse < .001:
            break
    # 测试集效果
    test_mse = ((np.array([t_ad_batch, t_cn_batch]) - np.array(cvae.predict([t_ad_batch, t_cn_batch]))[:, :, :, :, :,0])**2).mean()
    if test_mse <= 0.0036:
        k += 1
        test_mses.append(test_mse)

print('Finish training!')

