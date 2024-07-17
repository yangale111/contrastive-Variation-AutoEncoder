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


result_dir = './MCI_AD_result/'
fake_dir = './Dataset/fake_data_MCI_AD/'
_ = os.mkdir(fake_dir) if not os.path.exists(fake_dir) else 0


MCI_dataset_filepath = sorted(glob('/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Dataset/spm_MCIdata/*'))
AD_dataset_filepath = sorted(glob('/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Dataset/spm_ADdata/*'))


MCI_flods = np.load('./Dataset/MCI_AD_MCI_flods.npy',allow_pickle = True)
AD_flods = np.load('./Dataset/MCI_AD_AD_flods.npy',allow_pickle = True)


latent_dim = 256
batch_size = 16
beta = 1
gamma = 100
disentangle = True
loss = list()
nbatches = 1000
cvae, z_encoder, s_encoder, cvae_decoder = get_MRI_CVAE_3D(latent_dim=latent_dim, beta=beta,disentangle=disentangle,
                                                           gamma=gamma, bias=True, batch_size=batch_size,
                                                           learning_rate=0.001)


def create_fake_dataset(z_encoder, s_encoder, train_data, batch_size, k):
    # 获得特征分布
    z_features = []
    s_features = []
    print(train_data.shape[0]//batch_size)
    for i in range(train_data.shape[0]//batch_size):
        z_features.append(z_encoder.predict(train_data[i*batch_size:(i+1)*batch_size])[2])
        s_features.append(s_encoder.predict(train_data[i*batch_size:(i+1)*batch_size])[2])
    z_features = np.array(z_features).reshape([-1,256])
    s_features = np.array(s_features).reshape([-1,256])
    z_features += np.random.normal(0, 1,z_features.shape)
    s_features += np.random.normal(0, 1,s_features.shape)
    print(z_features.shape)
    print(s_features.shape)

    mask = ants.image_read('./Template/MNI152_T1_2mm_brain.nii.gz')
    spacing = mask.spacing
    origin = mask.origin
    direction = mask.direction
    # 用这些参数来重采样第一个图像

    def create_fake_data(z_features, s_features):
        fake_data = cvae_decoder.predict(np.hstack((z_features, s_features))).reshape((91,109,91))
        # print(fake_data.shape)
        fake_data_a = ants.from_numpy(fake_data)
        ants.set_spacing(fake_data_a,spacing)
        ants.set_origin(fake_data_a,origin)
        ants.set_direction(fake_data_a,direction)
        registration_params = ants.registration(mask, fake_data_a, type_of_transform="Rigid")
        fake_data = ants.apply_transforms(mask, fake_data_a, transformlist=registration_params['fwdtransforms']).numpy()
        return np.reshape(fake_data, (91, 109, 91))

    fake_train_dir = fake_dir+'train_{}/'.format(k)
    _ = os.mkdir(fake_train_dir) if not os.path.exists(fake_train_dir) else 0

    fake_background_dir = fake_train_dir+'background/'
    fake_target_dir = fake_train_dir+'target/'
    _ = os.mkdir(fake_background_dir) if not os.path.exists(fake_background_dir) else 0
    _ = os.mkdir(fake_target_dir) if not os.path.exists(fake_target_dir) else 0

    fake_background_data = []
    fake_target_data = []
    for i in tqdm(range(z_features.shape[0])):
        z_vector = z_features[i].reshape((1,256))
        s_vector = s_features[i].reshape((1,256))
        zero_vector = np.zeros((1,256))
        fake_data = create_fake_data(z_vector, zero_vector)
        fake_background_data.append(fake_data)
        np.save(fake_background_dir+'{}_0.npy'.format(i), fake_data)
        fake_data = create_fake_data(z_vector,s_vector)
        fake_target_data.append(fake_data)
        np.save(fake_target_dir+'{}_1.npy'.format(i), fake_data)
    return np.array(fake_background_data), np.array(fake_target_data)


k = 0
# fake_data_nums = 100
while k < 5:
    # 设置参数
    # 模型参数保存路径
    fn = result_dir + 'tf_weights_mae_MCI_AD_{}_{}_{}/'.format(beta, gamma, k)
    # 加载数据路径
    MCI_trainset_filepath = [MCI_dataset_filepath[i] for i in MCI_flods[k][0]]
    MCI_testset_filepath = [MCI_dataset_filepath[i] for i in MCI_flods[k][1]]
    AD_trainset_filepath = [AD_dataset_filepath[i] for i in AD_flods[k][0]]
    AD_testset_filepath = [AD_dataset_filepath[i] for i in AD_flods[k][1]]
    # 读取数据
    MCI_train_data = load_data(MCI_trainset_filepath)
    MCI_test_data = load_data(MCI_testset_filepath)
    AD_train_data = load_data(AD_trainset_filepath)
    AD_test_data = load_data(AD_testset_filepath)
    print('第{}折数据增强'.format(k))

    z_encoder.load_weights(fn + 'z_encoder.h5')
    s_encoder.load_weights(fn + 's_encoder.h5')
    cvae_decoder.load_weights(fn + 'cvae_decoder.h5')

    AD_train_data_nums = AD_train_data.shape[0]
    MCI_train_data_nums = MCI_train_data.shape[0]
    fake_background_data, fake_target_data = create_fake_dataset(z_encoder, s_encoder, AD_train_data, batch_size, k)

    create_tfrecord(MCI_train_data, AD_train_data, fake_background_data, fake_target_data, fake_dir+'train_{}/'.format(k) + 'aug_train.tfrecord')
    create_tfrecord(MCI_train_data, AD_train_data,[],[], fake_dir+'train_{}/'.format(k) + 'ori_train.tfrecord')
    create_tfrecord(MCI_test_data, AD_test_data, [],[], fake_dir+'train_{}/'.format(k) + 'test.tfrecord')

    k+=1
