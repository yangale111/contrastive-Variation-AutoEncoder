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
from scipy.ndimage import zoom
from skimage.transform import resize


def grad_cam_plus(model, img, layer_name="block5_conv3", label_name=None, category_id=None):
    img_tensor = img
    conv_layer = model.get_layer(layer_name)
    heatmap_model = Model([model.inputs], [conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        inputs = tf.cast(img_tensor, tf.float32)
        conv_output, predictions = heatmap_model(img_tensor)
        if category_id is None:
            category_id = np.argmax(predictions[0])
        if label_name:
            print(label_name[category_id])
        output = predictions[:, category_id]

    # compute the gradient of the score for the class c, with respect to feature maps Ak of a convolutional layer
    batch_grads = tape.gradient(output, conv_output)
    grads = batch_grads[0]
    conv_first_grad = tf.exp(output) * grads
    conv_second_grad = tf.exp(output) * tf.pow(grads, 2)
    conv_third_grad = tf.exp(output) * tf.pow(grads, 3)

    # print(conv_first_grad.shape)

    global_sum = tf.reduce_sum(tf.reshape(conv_output[0], shape=(-1, conv_output[0].shape[3])), axis=0)
    alpha_num = conv_second_grad
    alpha_denom = conv_second_grad * 2.0 + conv_third_grad * tf.reshape(global_sum,
                                                                        shape=(1, 1, 1, conv_output[0].shape[3]))
    alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones(shape=alpha_denom.shape))
    alphas = alpha_num / (alpha_denom)
    weights = tf.maximum(conv_first_grad, 0.0)
    alpha_normalization_constant = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(alphas, axis=0), axis=0), axis=0)
    alphas /= tf.reshape(alpha_normalization_constant, shape=(1, 1, 1, conv_second_grad.shape[3]))
    alphas_thresholding = np.where(weights, alphas, 0.0)

    alpha_normalization_constant = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(alphas_thresholding, axis=0), axis=0),axis=0)
    alpha_normalization_constant_processed = tf.where(alpha_normalization_constant != 0.0, alpha_normalization_constant,tf.ones(alpha_normalization_constant.shape))

    alphas /= tf.reshape(alpha_normalization_constant_processed, shape=(1, 1, 1, conv_second_grad.shape[3]))
    deep_linearization_weights = tf.reduce_sum(tf.reshape((weights * alphas), shape=(-1, conv_second_grad.shape[3])),axis=0)
    grad_CAM_map = tf.reduce_sum(deep_linearization_weights * conv_output[0], axis=3)
    cam = np.maximum(grad_CAM_map, 0)
    heatmap = cam / np.max(cam)
    heatmap = np.squeeze(heatmap)
    attMap = np.zeros_like(img)
    zoom_factor = (91 / heatmap.shape[0], 109 / heatmap.shape[1], 91 / heatmap.shape[2])
    attMap[..., 0] = zoom(heatmap, zoom_factor)
    # attMap = np.maximum(attMap, 0)  # 相当于一个relu
    attMap = np.squeeze(attMap)
    # attMap = np.maximum(attMap, 0)
    # attMap = attMap / np.max(attMap)
    return attMap, predictions.numpy()[0][0]


class GradCAMPlusPlus3D:
    def __init__(self, model, class_idx, layer_name=None):
        self.model = model
        self.class_idx = class_idx
        self.layer_name = layer_name
        if not layer_name:
            self.layer_name = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 5:  # 3D数据通常有5个维度
                return layer.name
        raise ValueError('Could not find 3D layer. Cannot apply GradCAMPlusPlus3D.')

    def compute_cam(self, volume, eps=1e-8):
        # 创建用于计算梯度的模型
        grad_model = tf.keras.models.Model(inputs=[self.model.inputs],
                                          outputs=[self.model.get_layer(self.layer_name).output, self.model.output])
        with tf.GradientTape() as tape:
            inputs = tf.cast(volume, tf.float32)
            conv_outs, predictions = grad_model(inputs)
            y_c = predictions[:, self.class_idx]
        # 计算梯度
        batch_grads = tape.gradient(y_c, conv_outs)
        grads = batch_grads[0]
        first = tf.exp(y_c) * grads
        second = tf.exp(y_c) * tf.pow(grads, 2)
        third = tf.exp(y_c) * tf.pow(grads, 3)
        # 计算权重
        global_sum = tf.reduce_sum(tf.reshape(conv_outs[0], shape=(-1, first.shape[3])), axis=0)
        alpha_num = second
        alpha_denom = second * 2.0 + third * tf.reshape(global_sum, shape=(1,1,1, first.shape[3]))
        alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones(shape=alpha_denom.shape))
        alphas = alpha_num / (alpha_denom + eps)
        weights = tf.maximum(first, 0.0)
        alpha_normalization_constant = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(alphas, axis=0),axis=0),axis=0)
        alphas /= (tf.reshape(alpha_normalization_constant, shape=(1,1,1,-1)) + eps)
        # print(alphas.shape, alpha_normalization_constant.shape)
        alphas_thresholding = tf.where(weights > 0.0, alphas, 0.0)

        alpha_normalization_constant = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(alphas_thresholding, axis=0), axis=0),axis=0)
        alpha_normalization_constant_processed = tf.where(alpha_normalization_constant != 0.0, alpha_normalization_constant,
                                                          tf.ones(alpha_normalization_constant.shape))
        alphas /= (tf.reshape(alpha_normalization_constant_processed, shape=(1,1,1, first.shape[3]))+eps)
        deep_linearization_weights = tf.reduce_sum(tf.reshape(weights * alphas, shape=(-1, first.shape[3])), axis=0)
        grad_CAM_map = tf.reduce_sum(deep_linearization_weights * conv_outs[0], axis=3)
        cam = tf.maximum(grad_CAM_map, 0)
        cam = cam / (tf.reduce_max(cam) + eps)  # 归一化
        cam = resize(cam,(91,109,91))

        return cam


k = 0
# fake_data_nums = 100
attMAPs = []
cnn_weights='/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/result_1/tf_weights_mae_CN_MCI_0/model_63-0.89.h5'
learning_rate = 0.0001
model = Mynet2(learning_rate)
model.load_weights(cnn_weights)
grad_cam_plus_plus = GradCAMPlusPlus3D(model, 0, layer_name='conv5')
while k < 1:
    print(k)
    test_dataset = load_tf_data(fake_dir+'train_{}/'.format(k) + 'test.tfrecord',1,1,False)
    model.evaluate(test_dataset,batch_size=1)
    for x, y in tqdm(test_dataset):
        attMap = grad_cam_plus_plus.compute_cam(x)
        attMAPs.append(np.nan_to_num(attMap))
    k+=1


attMAPs = np.array(attMAPs)
mean_attmap = np.mean(attMAPs,axis=0)


def get_mask_corr_nii(str,corr_nii):
    mask = ants.image_read('/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/GM_mask.nii.gz')
    mask_n = mask.numpy()
    corr_nii_mask = np.where(mask_n != 0, corr_nii, 0)
    # corr_nii_mask = corr_nii
    def max_min_nor(data):
        non_zero_elements = data[data != 0]
        min_value = np.min(non_zero_elements)
        max_value = np.max(non_zero_elements)
        # 对非零元素进行最大最小值归一化
        data[data != 0] = (data[data != 0] - min_value) / (max_value - min_value)
        return data
    corr_nii_mask[corr_nii_mask  <  0.8 * np.max(corr_nii_mask)] = 0
    # corr_nii_mask = max_min_nor(corr_nii_mask)
    corr_nii_mask_a = ants.from_numpy(corr_nii_mask)
    spacing = mask.spacing
    origin = mask.origin
    direction = mask.direction
    # 用这些参数来重采样第一个图像
    ants.set_spacing(corr_nii_mask_a,spacing)
    ants.set_origin(corr_nii_mask_a,origin)
    ants.set_direction(corr_nii_mask_a,direction)
    ants.plot_ortho(mask,corr_nii_mask_a)
    ants.image_write(corr_nii_mask_a,str)
    return corr_nii_mask


salient_map = get_mask_corr_nii('./roi_result/CN_MCI_salient_map_3.nii.gz',mean_attmap)


def get_the_positive_areas(corr_nii_mask):
    aal2 = ants.image_read('/root/commonfiles/wangyuqi/Sex/CVAE_Predict_Brainsex_91_109_91/aal2.nii.gz')
    aal2_n = aal2.numpy()

    ycorr_nii_mask = corr_nii_mask
    areas = [0 for i in range(121)]
    print(len(areas))
    for i in range(1,121):
        aal2_i_v = ycorr_nii_mask[aal2_n == i]
        areas[i] = np.max(aal2_i_v)
        # areas[i] = t
    map = {}
    for i in range(len(areas)):
        area = areas[i]
        map[i] = area
    map = dict(sorted(map.items(), key = lambda map:map[1], reverse = True))
    print(map)
    values = map.values()
    x = list(values)
    arr = np.array(x)

    mean = np.mean(arr)
    std = np.std(arr)

    lower_threshold = mean - (2 * std)
    upper_threshold = mean + (2 * std)

    outliers = [x for x in arr if x < lower_threshold or x > upper_threshold]
    print(outliers)


get_the_positive_areas(salient_map)


def get_top_salient_map(salient_map, top_value,save_str):
    aal2 = ants.image_read('/root/commonfiles/wangyuqi/Sex/CVAE_Predict_Brainsex_91_109_91/aal2.nii.gz')
    aal2_n = aal2.numpy()
    salient_map_top = salient_map
    values_to_find = top_value
    # 创建一个布尔掩码数组，指示A中与values_to_find匹配的位置
    mask = np.isin(aal2_n, values_to_find)
    # 使用布尔掩码将B中非positions的值设为0
    salient_map_top[~mask] = 0
    salient_map_top_a = ants.from_numpy(salient_map_top)
    spacing = aal2.spacing
    origin = aal2.origin
    direction = aal2.direction
    # 用这些参数来重采样第一个图像
    ants.set_spacing(salient_map_top_a,spacing)
    ants.set_origin(salient_map_top_a,origin)
    ants.set_direction(salient_map_top_a,direction)
    ants.plot_ortho(aal2,salient_map_top_a)
    ants.image_write(salient_map_top_a,save_str)


get_top_salient_map(salient_map,[43,41,45,17],'./roi_result/CN_MCI_top_salient_map.nii.gz')
