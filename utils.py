from glob import glob
import ants
from scipy.special import km
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
from sklearn.model_selection import KFold
from utils import *

CN_dataset_filepath = sorted(glob('/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Dataset/spm_CNdata/*'))
AD_dataset_filepath = sorted(glob('/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Dataset/spm_ADdata/*'))


def split_dataset():
    kf = KFold(n_splits=5,shuffle=True)
    kf.get_n_splits(AD_dataset_filepath)
    flods = []
    for i, (train_index, test_index) in enumerate(kf.split(AD_dataset_filepath)):
         print(f"Fold {i}:")
         print(f"  Train: index={train_index}")
         print(f"  Test:  index={test_index}")
         flods.append([train_index,test_index])
    flods = np.array(flods)
    np.save('./Dataset/CN_AD_AD_flods.npy',flods)

    kf = KFold(n_splits=5,shuffle=True)
    kf.get_n_splits(CN_dataset_filepath)
    flods = []
    for i, (train_index, test_index) in enumerate(kf.split(CN_dataset_filepath)):
         print(f"Fold {i}:")
         print(f"  Train: index={train_index}")
         print(f"  Test:  index={test_index}")
         flods.append([train_index,test_index])
    flods = np.array(flods)
    np.save('./Dataset/CN_AD_CN_flods.npy',flods)

def load_data(file_path):
    train_data = []
    for path in file_path:
        train_data.append(ants.image_read(path).numpy())
    train_data = np.array(train_data)
    return train_data


def plot_four(TG_batch, BG_batch, z_encoder, s_encoder, cvae_decoder, cvae, idx=0, v=2, s=0, k=40, axis='ax'):
    im_in = [TG_batch, BG_batch][idx]  # idx是0 模型输入就是tg ， 是1就是bg
    _zeros = np.zeros(s_encoder(im_in)[2].shape)
    # print(s_encoder(im_in)[2].shape)
    cvae_sal_vec = np.hstack((_zeros, s_encoder(im_in)[v]))  # 单独输出 salient
    cvae_bg_vec = np.hstack((z_encoder(im_in)[v], _zeros))  # 单独输出 background

    if idx == 0:  # 如果是0 代表输入的是TG
        cvae_full_vec = np.hstack((z_encoder(im_in)[v], s_encoder(im_in)[v]))
    elif idx == 1:  # 如果是1 输入就是BG可以直接输出bg_vec
        cvae_full_vec = cvae_bg_vec

    if axis == 'ax':
        plot_im_input = im_in[s, :, :, k]
        plot_im_sal = cvae_decoder(cvae_sal_vec)[s, :, :, k, 0]
        plot_im_bg = cvae_decoder(cvae_bg_vec)[s, :, :, k, 0]
        plot_im_recon = cvae_decoder(cvae_full_vec)[s, :, :, k, 0]
    elif axis == 'sag':
        plot_im_input = im_in[s, k, :, :]
        plot_im_sal = cvae_decoder(cvae_sal_vec)[s, k, :, :, 0]
        plot_im_bg = cvae_decoder(cvae_bg_vec)[s, k, :, :, 0]
        plot_im_recon = cvae_decoder(cvae_full_vec)[s, k, :, :, 0]
    elif axis == 'cor':
        plot_im_input = im_in[s, :, k, :]
        plot_im_sal = cvae_decoder(cvae_sal_vec)[s, :, k, :, 0]
        plot_im_bg = cvae_decoder(cvae_bg_vec)[s, :, k, :, 0]
        plot_im_recon = cvae_decoder(cvae_full_vec)[s, :, k, :, 0]

    plt.figure(figsize=np.array((4 * 4, 4)) * .5)
    plt.subplot(1, 4, 1)
    plt.imshow(plot_im_input);
    plt.xticks([]);
    plt.yticks([]);
    plt.title('input')
    plt.subplot(1, 4, 2)
    plt.imshow(plot_im_recon);
    plt.xticks([]);
    plt.yticks([]);
    plt.title('reconstruction')
    plt.subplot(1, 4, 3)
    plt.imshow(plot_im_sal);
    plt.xticks([]);
    plt.yticks([]);
    plt.title('salient')
    plt.subplot(1, 4, 4)
    plt.imshow(plot_im_bg);
    plt.xticks([]);
    plt.yticks([]);
    plt.title('background')

    plt.show()
def cvae_query(ABIDE_data, s_encoder, z_encoder, cvae_decoder):
    i = 0
    n = 8
    v_sl = s_encoder.predict(ABIDE_data[0:n, :, :, :])[i]  # [0,:]
    v_bg = z_encoder.predict(ABIDE_data[0:n, :, :, :])[i]  # [0,:]
    v = np.hstack((v_bg, v_sl))
    latent_vec = v
    out = cvae_decoder.predict(latent_vec)

    im = out[:, :, :, :, 0]
    im1 = ABIDE_data[0:n, :, :, :]
    ss = ((im - im1) ** 2).sum()

    return im[0, :, :, 40], im1[0, :, :, 40], ss


class ChannelAttention(layers.Layer):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg = layers.GlobalAveragePooling3D()
        self.max = layers.GlobalMaxPooling3D()
        self.conv1 = layers.Conv3D(in_planes // ratio, kernel_size=1, strides=1, padding='same',
                                   kernel_regularizer=regularizers.l2(5e-4),
                                   use_bias=True, activation=tf.nn.relu)
        self.conv2 = layers.Conv3D(in_planes, kernel_size=1, strides=1, padding='same',
                                   kernel_regularizer=regularizers.l2(5e-4),
                                   use_bias=True)

    def call(self, inputs):
        avg = self.avg(inputs)
        max = self.max(inputs)
        avg = layers.Reshape((1, 1,1, avg.shape[1]))(avg)  # shape (None, 1, 1 feature)
        max = layers.Reshape((1, 1,1, max.shape[1]))(max)  # shape (None, 1, 1 feature)
        avg_out = self.conv2(self.conv1(avg))
        max_out = self.conv2(self.conv1(max))
        out = avg_out + max_out
        out = tf.nn.sigmoid(out)

        return out

class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = Conv3D(filters=1,
                           kernel_size=kernel_size,
                           strides=1,
                           padding='same',
                           activation='sigmoid',
                           kernel_initializer='he_normal',
                           use_bias=False)

    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=4)
        max_out = tf.reduce_max(inputs, axis=4)
        out = tf.stack([avg_out, max_out], axis=4)  # 创建一个维度,拼接到一起concat。
        out = self.conv1(out)

        return out

class CBAM(layers.Layer):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channel)
        # 通道注意力
        self.sa = SpatialAttention()
        # 空间注意力
    def call(self, inputs):
        out = self.ca(inputs) * inputs
        out = self.sa(out) * out
        return out


def sampling(args):

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def get_MRI_CVAE_3D(input_shape=(91, 109, 91, 1), latent_dim=2, beta=1, disentangle=False, gamma=1, bias=True,batch_size=64,learning_rate = 0.001):
    image_size, _, _, channels = input_shape
    kernel_size = 3
    filters1 = [16,32,64,128]
    filters2 = [128,64,32,16]
    intermediate_dim = 2*latent_dim

    # build encoder model
    tg_inputs = Input(shape=input_shape, name='tg_inputs')  # tg_input是对比的输入
    bg_inputs = Input(shape=input_shape, name='bg_inputs')  # bg_input是参照输入


    z_h_layer = Dense(intermediate_dim, activation='relu', use_bias=bias)
    z_mean_layer = Dense(latent_dim, name='z_mean', use_bias=bias)
    z_log_var_layer = Dense(latent_dim, name='z_log_var', use_bias=bias)
    z_layer = Lambda(sampling, output_shape=(latent_dim,), name='z')

    z_conv1 = Conv3D(filters=filters1[0],
                 kernel_size=(4,6,4),
                 strides=2,
                 padding='valid',
                 activation='relu')
    z_cbam1 = CBAM(filters1[0])
    z_conv2 = Conv3D(filters=filters1[1],
                 kernel_size=3,
                 strides=2,
                 padding='same',
                 activation='relu')
    z_cbam2 = CBAM(filters1[1])
    z_conv3 = Conv3D(filters=filters1[2],
                 kernel_size=3,
                 strides=2,
                 padding='valid',
                 activation='relu')
    z_cbam3 = CBAM(filters1[2])
    z_conv4 = Conv3D(filters=filters1[3],
                 kernel_size=3,
                 strides=2,
                 padding='same',
                 activation='relu')
    z_cbam4 = CBAM(filters1[3])
    def z_encoder_func(inputs):
        z_h = inputs
        # for i in range(nlayers):
        z_h = z_conv1(z_h)
        z_h = z_cbam1(z_h)

        z_h = z_conv2(z_h)
        z_h = z_cbam2(z_h)

        z_h = z_conv3(z_h)
        z_h = z_cbam3(z_h)

        z_h = z_conv4(z_h)
        z_h = z_cbam4(z_h)

        shape = K.int_shape(z_h)
        z_h = Flatten()(z_h)
        z_h = z_h_layer(z_h)
        z_mean = z_mean_layer(z_h)
        z_log_var = z_log_var_layer(z_h)
        z = z_layer([z_mean, z_log_var])
        return z_mean, z_log_var, z, shape

    tg_z_mean, tg_z_log_var, tg_z, shape_z = z_encoder_func(tg_inputs)
    bg_z_mean, bg_z_log_var, bg_z, _ = z_encoder_func(bg_inputs)


    # generate latent vector Q(z|X)
    s_h_layer = Dense(intermediate_dim, activation='relu', use_bias=bias)
    s_mean_layer = Dense(latent_dim, name='s_mean', use_bias=bias)
    s_log_var_layer = Dense(latent_dim, name='s_log_var', use_bias=bias)
    s_layer = Lambda(sampling, output_shape=(latent_dim,), name='s')

    s_conv1 = Conv3D(filters=filters1[0],
                 kernel_size=(4,6,4),
                 strides=2,
                 padding='valid',
                 activation='relu')
    s_cbam1 = CBAM(filters1[0])
    s_conv2 = Conv3D(filters=filters1[1],
                 kernel_size=3,
                 strides=2,
                 padding='same',
                 activation='relu')
    s_cbam2 = CBAM(filters1[1])
    s_conv3 = Conv3D(filters=filters1[2],
                 kernel_size=3,
                 strides=2,
                 padding='valid',
                 activation='relu')
    s_cbam3 = CBAM(filters1[2])
    s_conv4 = Conv3D(filters=filters1[3],
                 kernel_size=3,
                 strides=2,
                 padding='same',
                 activation='relu')
    s_cbam4 = CBAM(filters1[3])

    def s_encoder_func(inputs):
        s_h = inputs
        s_h = s_conv1(s_h)
        s_h = s_cbam1(s_h)
        s_h = s_conv2(s_h)
        s_h = s_cbam2(s_h)
        s_h = s_conv3(s_h)
        s_h = s_cbam3(s_h)
        s_h = s_conv4(s_h)
        s_h = s_cbam4(s_h)

        # shape info needed to build decoder model
        shape = K.int_shape(s_h)
        s_h = Flatten()(s_h)
        s_h = s_h_layer(s_h)
        s_mean = s_mean_layer(s_h)
        s_log_var = s_log_var_layer(s_h)
        s = s_layer([s_mean, s_log_var])
        return s_mean, s_log_var, s, shape

    tg_s_mean, tg_s_log_var, tg_s, shape_s = s_encoder_func(tg_inputs)

    # instantiate encoder models
    z_encoder = tf.keras.models.Model(tg_inputs, [tg_z_mean, tg_z_log_var, tg_z], name='z_encoder')
    s_encoder = tf.keras.models.Model(tg_inputs, [tg_s_mean, tg_s_log_var, tg_s], name='s_encoder')

    # build decoder model
    latent_inputs = Input(shape=(2 * latent_dim,), name='z_sampling')

    x = Dense(intermediate_dim, activation=gelu, use_bias=bias)(latent_inputs)
    x = Dense(shape_z[1] * shape_z[2] * shape_z[3] * shape_z[4], activation=gelu, use_bias=bias)(x)
    x = Reshape((shape_z[1], shape_z[2], shape_z[3], shape_z[4]))(x)

    # for i in range(nlayers):
    # x = transconv_batch_and_activation(x,filters2[0],3,2,1,'same')
    x = Conv3DTranspose(filters=filters2[0],
                        kernel_size=3,
                        strides=2,
                        padding='same',
                        output_padding=1,
                        activation = 'relu')(x)
    x = Conv3DTranspose(filters=filters2[1],
                        kernel_size=3,
                        strides=2,
                        padding='valid',
                        output_padding=1,
                        activation = 'relu')(x)
    x = Conv3DTranspose(filters=filters2[2],
                        kernel_size=3,
                        strides=2,
                        padding='same',
                        output_padding=1,
                        activation = 'relu')(x)
    x = Conv3DTranspose(filters=filters2[3],
                        kernel_size=(4,6,4),
                        strides=2,
                        padding='valid',
                        output_padding=1,
                        activation = 'relu')(x)

    outputs = Conv3DTranspose(filters=1,
                              kernel_size=1,
                              activation='relu',
                              padding='same',
                              use_bias=bias,
                              name='decoder_output')(x)

    # instantiate decoder model
    cvae_decoder = Model(latent_inputs, outputs, name='decoder')

    def zeros_like(x):
        return tf.zeros_like(x)

    tg_outputs = cvae_decoder(tf.keras.layers.concatenate([tg_z, tg_s], -1))
    zeros = tf.keras.layers.Lambda(zeros_like)(tg_s)

    bg_outputs = cvae_decoder(tf.keras.layers.concatenate([bg_z, zeros], -1))

    # instantiate VAE model
    cvae = tf.keras.models.Model(inputs=[tg_inputs, bg_inputs],
                                 outputs=[tg_outputs, bg_outputs],
                                 name='contrastive_vae')

    if disentangle:
        discriminator = Dense(1, activation='sigmoid')

        z1 = Lambda(lambda x: x[:int(batch_size / 2), :])(tg_z)
        z2 = Lambda(lambda x: x[int(batch_size / 2):, :])(tg_z)
        s1 = Lambda(lambda x: x[:int(batch_size / 2), :])(tg_s)
        s2 = Lambda(lambda x: x[int(batch_size / 2):, :])(tg_s)

        q_bar = tf.keras.layers.concatenate(
            [tf.keras.layers.concatenate([s1, z2], axis=1),
             tf.keras.layers.concatenate([s2, z1], axis=1)],
            axis=0)

        q = tf.keras.layers.concatenate(
            [tf.keras.layers.concatenate([s1, z1], axis=1),
             tf.keras.layers.concatenate([s2, z2], axis=1)],
            axis=0)

        q_bar_score = (discriminator(q_bar) + .1) * .85  # +.1 * .85 so that it's 0<x<1
        q_score = (discriminator(q) + .1) * .85
        tc_loss = K.log(q_score / (1 - q_score))
        discriminator_loss = - K.log(q_score) - K.log(1 - q_bar_score)
    else:
        tc_loss = 0
        discriminator_loss = 0

    reconstruction_loss = tf.keras.losses.mse(K.flatten(tg_inputs), K.flatten(tg_outputs))
    reconstruction_loss += tf.keras.losses.mse(K.flatten(bg_inputs), K.flatten(bg_outputs))
    reconstruction_loss *= input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]

    kl_loss = 1 + tg_z_log_var - tf.keras.backend.square(tg_z_mean) - tf.keras.backend.exp(tg_z_log_var)
    kl_loss += 1 + tg_s_log_var - tf.keras.backend.square(tg_s_mean) - tf.keras.backend.exp(tg_s_log_var)
    kl_loss += 1 + bg_z_log_var - tf.keras.backend.square(bg_z_mean) - tf.keras.backend.exp(bg_z_log_var)
    kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    cvae_loss = tf.keras.backend.mean(reconstruction_loss + beta * kl_loss + gamma * tc_loss + discriminator_loss)
    cvae.add_loss(cvae_loss)

    cvae.add_metric(reconstruction_loss,'reconstruction_loss')
    cvae.add_metric(kl_loss,'kl_loss')
    cvae.add_metric(tc_loss,'tc_loss')
    cvae.add_metric(discriminator_loss,'discriminator_loss')
    # opt = Lion(learning_rate = learning_rate)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate,beta_1=0.9,beta_2=0.999,epsilon=1e-07,amsgrad=False,name='Adam')

    cvae.compile(optimizer=opt, run_eagerly=True)

    return cvae, z_encoder, s_encoder, cvae_decoder


def get_recon_and_twin(inMat,z_encoder,s_encoder,cvae_decoder):
    z = z_encoder.predict(inMat)[2]
    s = s_encoder.predict(inMat)[2]
    zeros = np.zeros(s.shape)

    recon = cvae_decoder.predict(np.hstack((z,s)))[:,:,:,:,0]
    twin = cvae_decoder.predict(np.hstack((z,zeros)))[:,:,:,:,0]

    return recon,twin
def get_Js(invec, inMat, recon, twin, template, ofdir):

    interpolator = 'bSpline'

    for i in tqdm(range(len(invec))):
        sub = inMat[i]
        t1 = ants.from_numpy(sub)


        recon_mat = recon[i,:,:,:]
        twin_mat = twin[i,:,:,:]
        # print(twin_mat.shape)
        recon_brain = t1.new_image_like(recon_mat)
        # recon_mat 是一个三维的 Numpy 数组，表示需要进行配准的参考图像。使用 new_image_like 方法可以方便地创建一个与 recon_mat 维度相同、像素间距与 t1 相同的图像对象，便于之后的图像配准操作。
        twin_brain = t1.new_image_like(twin_mat)

        recon_brain = ants.iMath_normalize(recon_brain)
        twin_brain = ants.iMath_normalize(twin_brain)

        # Match twin to recon
        # twin_brain = ants.registration(fixed=recon_brain,moving=twin_brain,type_of_transform='Rigid')['warpedmovout']
        tx2t1 = ants.registration(fixed=t1,moving=recon_brain,type_of_transform='Rigid')
        # print(tx2t1)
        # Match twin and recon to T1
        recon_brain = ants.apply_transforms(fixed=t1,moving=recon_brain,transformlist=tx2t1['fwdtransforms'],interpolator=interpolator)
        twin_brain = ants.apply_transforms(fixed=t1,moving=twin_brain,transformlist=tx2t1['fwdtransforms'],interpolator=interpolator)

        # calculate jacobian in native space
        tx = ants.registration(fixed=recon_brain,moving=twin_brain,type_of_transform='SyN')
        J = ants.create_jacobian_determinant_image(domain_image=twin_brain,tx=tx['fwdtransforms'][0])
        J = J-1

        norm = ants.registration(fixed=template,moving=t1,type_of_transform='SyN')
        normed_t1 = ants.apply_transforms(fixed=template,moving=t1,transformlist=norm['fwdtransforms'],interpolator=interpolator)

        normed_recon = ants.apply_transforms(fixed=template,moving=recon_brain,transformlist=norm['fwdtransforms'],interpolator=interpolator)
        normed_twin = ants.apply_transforms(fixed=template,moving=twin_brain,transformlist=norm['fwdtransforms'],interpolator=interpolator)
        normed_J = ants.apply_transforms(fixed=template,moving=J,transformlist=norm['fwdtransforms'],interpolator=interpolator)

        # SAVE THE RESULTS

        # Where to save everything
        # ofdir = '/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Jacobian_CN_MCIAD_{}/'.format(k)

        # MAKE OFDIR IF NOT EXIST
        _ = os.mkdir(ofdir) if not os.path.exists(ofdir) else 0

        # Make a dict
        res = dict()

        res['native_Js'] = J
        res['twin_brains'] = twin_brain
        res['recon_brains'] = recon_brain
        res['t1s'] = t1

        res['normed_Js'] = normed_J
        res['normed_t1s'] = normed_t1
        res['normed_recons'] = normed_recon
        res['normed_twins'] = normed_twin

        for key in list(res.keys()):
            # make a subdir if needed
            _ = os.mkdir(os.path.join(ofdir,key)) if os.path.exists(os.path.join(ofdir,key))==False else 0 # One liner if statement
            res[key].to_filename(os.path.join(ofdir,key,f'{i}_{key}.nii'))

    return res


# 将背景数据、目标数据以及生成的假数据和假目标数据写入TFRecord文件
def create_tfrecord(background_data, target_data, fake_background_data,fake_target_data, savePath):
    writer = tf.compat.v1.python_io.TFRecordWriter(savePath)
        # just zoomed data

    def write_data(data, label):
        for d in data:
            img_raw = d.tobytes()
            # print(type(label))
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            n = writer.write(example.SerializeToString())
    write_data(background_data, 0.0)
    write_data(target_data, 1.0)
    write_data(fake_background_data, 0.0)
    write_data(fake_target_data, 1.0)
    writer.close()
features = {"label": tf.compat.v1.FixedLenFeature([1], tf.float32),
            "img_raw": tf.compat.v1.FixedLenFeature([], tf.string)}

# 用于解析TFRecord文件中的单个样本
def _parse_image(example_proto):
    parsed_features = tf.compat.v1.parse_single_example(example_proto, features)
    img = tf.compat.v1.decode_raw(parsed_features['img_raw'], tf.float32)
    # img = tf.reshape(img, [110, 120, 110, 1])
    img = tf.reshape(img, [91, 109, 91, 1])
    img = tf.cast(img, tf.float32)  # 为什么又要来一次tf.float32
    label = tf.cast(parsed_features['label'], tf.int32)
    label = tf.reshape(label, [1])
    # label = tf.one_hot(label,2)
    # label = tf.reshape(label,[2])
    return img, label

# 加载TFRecord数据集，并进行预处理
def load_tf_data(filename, batch_size, shuffle_buffer=None, repeat=True):
    dataset = tf.data.TFRecordDataset(filename)
    _parse_function = _parse_image
    dataset = dataset.map(_parse_function)  # 对每个dataset的每个样本调用_parse_function来读取TFRecords数据
    if shuffle_buffer is not None:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)  #
    if batch_size !=0:
        dataset = dataset.prefetch(batch_size)
    if repeat:
        dataset = dataset.repeat()
    if batch_size !=0:
        dataset = dataset.batch(batch_size)
    return dataset


def atrous_spatial_pyramid_pooling(inputs, i, filters=256, dilations=[1, 2, 4], regularizer=None):  # ASPP层
    '''
    Atrous Spatial Pyramid Pooling (ASPP) Block
    '''
    resize_height, resize_width, resize_length = inputs.shape[1], inputs.shape[2], inputs.shape[3]
    # print(resize_width)
    # Atrous Spatial Pyramid Pooling
    # Atrous 1x1
    aspp1x1 = Conv3D(filters=filters, kernel_size=1,
                     padding='same', kernel_regularizer=regularizer,
                     name='aspp1x1_' + i, activation='relu')(inputs)
    # K = k + (k−1)∗(r−1)
    # 3+2*(r-1)
    aspp3x3_1 = Conv3D(filters=filters, kernel_size=3,
                       padding='same', dilation_rate=dilations[0], kernel_regularizer=regularizer,
                       name='aspp3x3_1_' + i, activation='relu')(inputs)

    aspp3x3_2 = Conv3D(filters=filters, kernel_size=3,
                       padding='same', dilation_rate=dilations[1], kernel_regularizer=regularizer,
                       name='aspp3x3_2_' + i, activation='relu')(inputs)

    aspp3x3_3 = Conv3D(filters=filters, kernel_size=3,
                       padding='same', dilation_rate=dilations[2], kernel_regularizer=regularizer,
                       name='aspp3x3_3_' + i, activation='relu')(inputs)

    # Image Level Pooling
    image_feature = tf.reduce_mean(inputs, [1, 2, 3], keepdims=True)
    image_feature = Conv3D(filters=filters, kernel_size=1, padding='same', activation='relu')(image_feature)

    # print(image_feature.shape)
    # image_feature = tf.reshape(image_feature,(-1,1,1,filters))
    image_feature = layers.UpSampling3D(size=(resize_height, resize_width, resize_length),
                                        name='image_pool_feature_' + i)(image_feature)
    # Merge Poolings
    outputs = tf.concat(values=[aspp1x1, aspp3x3_1, aspp3x3_2, aspp3x3_3, image_feature],
                        axis=4, name='aspp_pools_' + i)

    outputs = Conv3D(filters=filters, kernel_size=1,
                     padding='same', kernel_regularizer=regularizer, name='aspp_outputs_' + i, activation='relu')(
        outputs)
    return outputs


def Mynet(learning_rate):
    input = Input((91, 109, 91, 1))

    x = Conv3D(filters=8, kernel_size=3,strides = 2, padding='same', activation='relu',name = 'conv11')(input)
    x = Conv3D(filters=8, kernel_size=3,strides = 1, padding='same', activation='relu',name = 'conv12')(x)
    x = BatchNormalization(name = 'bn1')(x)

    x = Conv3D(filters=16, kernel_size=3,strides = 2, padding='same', activation='relu',name = 'conv21')(x)
    x = Conv3D(filters=16, kernel_size=3,strides = 1, padding='same', activation='relu',name = 'conv22')(x)
    x = Conv3D(filters=16, kernel_size=3,strides = 1, padding='same', activation='relu',name = 'conv23')(x)
    x = atrous_spatial_pyramid_pooling(x, '1', 16, [1, 2, 3])
    x = BatchNormalization(name = 'bn2')(x)

    x = Conv3D(filters=32, kernel_size=3,strides = 2, padding='same', activation='relu',name = 'conv31')(x)
    x = Conv3D(filters=32, kernel_size=3,strides = 1, padding='same', activation='relu',name = 'conv32')(x)
    x = atrous_spatial_pyramid_pooling(x, '2', 32, [1, 2, 3])
    x = BatchNormalization(name = 'bn3')(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = Model(input,x)

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate,beta_1=0.9,beta_2=0.999,epsilon=1e-07,amsgrad=False,name='Adam')
    # 编译模型
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy', km.precision(), km.recall(), km.f1_score()])
    # 打印模型结构
    # model.summary()
    return model


def Mynet2(learning_rate):
    filters = [16, 32, 64, 128, 256, 256, 512]
    input = Input((91, 109, 91, 1))
    x = Conv3D(filters[0], strides=2, kernel_size=3, name='conv1')(input)  # 45 54 45
    x = BatchNormalization(name='bn1')(x)
    x = LeakyReLU(name='relu1')(x)

    x = Conv3D(filters[1], strides=2, kernel_size=3, name='cov2')(x)  # 22 27 22
    x = BatchNormalization(name='bn2')(x)
    x = LeakyReLU(name='relu2')(x)
    # x = atrous_spatial_pyramid_pooling(x, '1', filters[2], [1, 3, 5])

    x = Conv3D(filters[2], strides=2, kernel_size=3, name='conv3')(x)  # 20 24 20
    x = BatchNormalization(name='bn3')(x)
    x = LeakyReLU(name='relu3')(x)
    x = atrous_spatial_pyramid_pooling(x, '1', filters[2], [1, 3, 5])

    x = Conv3D(filters[3], strides=1, kernel_size=3, name='conv4')(x)  # 10 12 10
    x = BatchNormalization(name='bn4')(x)
    x = LeakyReLU(name='relu4')(x)
    x = atrous_spatial_pyramid_pooling(x, '2', filters[3], [1, 2, 3])

    x = Conv3D(filters[4], strides=1, kernel_size=3, name='conv5')(x)  # 10 12 10
    x = BatchNormalization(name='bn5')(x)
    x = LeakyReLU(name='relu5')(x)
    x = atrous_spatial_pyramid_pooling(x, '3', filters[4], [1, 2, 3])


    x = Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = Dense(512, name="dense_0", activation=LeakyReLU())(x)
    x = Dense(128, name="dense_1", activation=LeakyReLU())(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate,beta_1=0.9,beta_2=0.999,epsilon=1e-07,amsgrad=False,name='Adam')
    # 编译模型
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy', km.precision(), km.recall(), km.f1_score()])
    # model.summary()
    return model
