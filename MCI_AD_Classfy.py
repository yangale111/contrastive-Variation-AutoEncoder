from utils import *
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
from CN_AD import *
from data_augmentation_CN_AD import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
from utils import *



MCI_dataset_filepath = sorted(glob('/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Dataset/spm_MCIdata/*'))
AD_dataset_filepath = sorted(glob('/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Dataset/spm_ADdata/*'))


MCI_flods = np.load('./Dataset/MCI_AD_MCI_flods.npy',allow_pickle = True)
AD_flods = np.load('./Dataset/MCI_AD_AD_flods.npy',allow_pickle = True)

# 增强数据
tf.get_logger().setLevel('ERROR')
for j in range(10):
    k = 0
    # fake_data_nums = 100
    while k < 5:
        # 设置参数
        # 模型参数保存路径
        print(j, ' ' , k)
        MCI_trainset_filepath = [MCI_dataset_filepath[i] for i in MCI_flods[k][0]]
        MCI_testset_filepath = [MCI_dataset_filepath[i] for i in MCI_flods[k][1]]
        AD_trainset_filepath = [AD_dataset_filepath[i] for i in AD_flods[k][0]]
        AD_testset_filepath = [AD_dataset_filepath[i] for i in AD_flods[k][1]]
        batch_size = 64
        epoch = 1000
        train_steps = (len(MCI_trainset_filepath)+len(AD_trainset_filepath))*2 // batch_size
        val_steps = (len(MCI_testset_filepath) + len(AD_testset_filepath))
        train_dataset = load_tf_data(fake_dir+'train_{}/'.format(k) + 'aug_train.tfrecord',batch_size,100,True)

        test_dataset = load_tf_data(fake_dir+'train_{}/'.format(k) + 'test.tfrecord',1,50,False)
        learning_rate = 0.0001
        model = Mynet(learning_rate)
        _ = os.mkdir('./result_1/tf_weights_mae_MCI_AD_{}/'.format(k)) if not os.path.exists('./result_1/tf_weights_mae_MCI_AD_{}/'.format(k)) else 0
        cnn_file_path='./result_1/tf_weights_mae_MCI_AD_{}/'.format(k)+"model_{epoch:02d}-{val_accuracy:.2f}.h5"
        callbacks_list = [
            callbacks.ReduceLROnPlateau(
             # This callback will monitor the validation loss of the model
             monitor='val_accuracy',
             factor=0.95,
             patience=3,
             verbose=1,
             min_lr=0.000001
            ),
            callbacks.ModelCheckpoint(
                filepath=cnn_file_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                save_weights_only=True,
                period=1,
            ),
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=150,
                verbose=1,
            )
        ]
        model.fit(train_dataset,batch_size = batch_size,
                  validation_data=test_dataset,validation_batch_size=1,
                  steps_per_epoch=train_steps,validation_steps=val_steps,shuffle=True,epochs=epoch,callbacks=callbacks_list)
        model.save_weights('./result_1/tf_weights_mae_MCI_AD_{}/'.format(k)+'model_{}.h5'.format(j))
        k+=1
        tf.compat.v1.reset_default_graph()



# 未增强数据
tf.get_logger().setLevel('ERROR')
k = 0
# fake_data_nums = 100
for j in range(10):
    k = 0
    while k < 5:
        # 设置参数
        # 模型参数保存路径
        MCI_trainset_filepath = [MCI_dataset_filepath[i] for i in MCI_flods[k][0]]
        MCI_testset_filepath = [MCI_dataset_filepath[i] for i in MCI_flods[k][1]]
        AD_trainset_filepath = [AD_dataset_filepath[i] for i in AD_flods[k][0]]
        AD_testset_filepath = [AD_dataset_filepath[i] for i in AD_flods[k][1]]
        print(len(MCI_trainset_filepath),len(AD_trainset_filepath))
        print(len(MCI_testset_filepath),len(AD_testset_filepath))
        batch_size = 64
        epoch = 200
        train_steps = (len(MCI_trainset_filepath)+len(AD_trainset_filepath))*1 // batch_size
        val_steps = (len(MCI_testset_filepath) + len(AD_testset_filepath))
        train_dataset = load_tf_data(fake_dir+'train_{}/'.format(k) + 'ori_train.tfrecord',batch_size,100,True)

        test_dataset = load_tf_data(fake_dir+'train_{}/'.format(k) + 'test.tfrecord',1,50,False)
        learning_rate = 0.0001
        # model = VGG16(learning_rate)
        model = Mynet(learning_rate)
        _ = os.mkdir('./result_1/tf_weights_mae_MCI_AD_ori_{}/'.format(k)) if not os.path.exists('./result_1/tf_weights_mae_MCI_AD_ori_{}/'.format(k)) else 0
        cnn_file_path='./result_1/tf_weights_mae_MCI_AD_ori_{}/'.format(k)+"model_{epoch:02d}-{val_accuracy:.2f}.h5"
        callbacks_list = [
            callbacks.ReduceLROnPlateau(
             # This callback will monitor the validation loss of the model
             monitor='accuracy',
             factor=0.95,
             patience=3,
             verbose=1,
             min_lr=0.000001,
            ),
            callbacks.ModelCheckpoint(
                filepath=cnn_file_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                save_weights_only=True,
                period=1,
            )
        ]
        model.fit(train_dataset,batch_size = batch_size,
                  validation_data=test_dataset,validation_batch_size=1,
                  steps_per_epoch=train_steps,validation_steps=val_steps,shuffle=True,epochs=epoch,callbacks=callbacks_list)
        tf.compat.v1.reset_default_graph()
        k+=1



tf.compat.v1.reset_default_graph()
test_dataset = load_tf_data(fake_dir + 'train_{}/'.format(k) + 'test.tfrecord', 1, 1, False)
model = Mynet(0.0001)
weights = sorted(glob('/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/result_1/tf_weights_mae_MCI_AD_ori_{}/*'.format(k)))
for weight in weights:
    p_y = []
    t_y = []
    model.load_weights(weight)
    for x, y in test_dataset:
        p_y.append(model.predict(x))
        t_y.append(y)
    p_y = np.array(p_y)
    t_y = np.array(t_y)

    p_y = p_y.reshape((-1))
    t_y = t_y.reshape((-1))
    predicted_labels = p_y
    test_labels = t_y
    predicted_labels = np.where(predicted_labels >= 0.5, 1, 0)
    # 计算准确率（ACC）
    # print(predicted_labels.shape)
    accuracy = accuracy_score(test_labels, predicted_labels)

    # 计算精确率（Precision）
    precision = precision_score(test_labels, predicted_labels)

    # 计算召回率/敏感性（Recall/Sensitivity）
    recall = recall_score(test_labels, predicted_labels)

    # 计算F1分数
    f1 = f1_score(test_labels, predicted_labels)
    print(accuracy,' ', precision,' ',recall,' ', f1)



p_y = []
t_y = []
k = 0
learning_rate = 0.0001
# model = VGG16(learning_rate)
model = Mynet(learning_rate)
while k < 5:
    cnn_weights=glob('./result_1/tf_weights_mae_MCI_AD_ori_{}/*'.format(k))
    MCI_testset_filepath = [MCI_dataset_filepath[i] for i in MCI_flods[k][1]]
    AD_testset_filepath = [AD_dataset_filepath[i] for i in AD_flods[k][1]]
    print(len(MCI_testset_filepath),len(AD_testset_filepath))
    val_steps = (len(MCI_testset_filepath) + len(AD_testset_filepath))
    test_dataset = load_tf_data(fake_dir+'train_{}/'.format(k) + 'test.tfrecord',1,50,False)

    model.load_weights(cnn_weights[0])
    for x, y in test_dataset:
        p_y.append(model.predict(x))
        t_y.append(y)
    k+=1

p_y = np.array(p_y)
t_y = np.array(t_y)

p_y = p_y.reshape((-1))
t_y = t_y.reshape((-1))

predicted_labels = p_y
test_labels = t_y



p_y_a = []
t_y_a = []
k = 0
while k < 5:
    cnn_weights=glob('./result_1/tf_weights_mae_MCI_AD_{}/*'.format(k))
    MCI_testset_filepath = [MCI_dataset_filepath[i] for i in MCI_flods[k][1]]
    AD_testset_filepath = [AD_dataset_filepath[i] for i in AD_flods[k][1]]
    print(len(MCI_testset_filepath),len(AD_testset_filepath))
    val_steps = (len(MCI_testset_filepath) + len(AD_testset_filepath))
    test_dataset = load_tf_data(fake_dir+'train_{}/'.format(k) + 'test.tfrecord',1,50,False)
    model.load_weights(cnn_weights[0])
    for x, y in test_dataset:
        p_y_a.append(model.predict(x))
        t_y_a.append(y)
    k+=1

p_y_a = np.array(p_y_a)
t_y_a = np.array(t_y_a)

p_y_a = p_y_a.reshape((-1))
t_y_a = t_y_a.reshape((-1))
# p_y = np.array(p_y).reshape((-1))
# print(p_y.shape)
predicted_labels_a = p_y_a
test_labels_a = t_y_a


def get_metrices(predicted_labels, test_labels):
    predicted_labels = np.where(predicted_labels >= 0.5, 1, 0)
    # 计算准确率（ACC）
    accuracy = accuracy_score(test_labels, predicted_labels)

    # 计算精确率（Precision）
    precision = precision_score(test_labels, predicted_labels)

    # 计算召回率/敏感性（Recall/Sensitivity）
    recall = recall_score(test_labels, predicted_labels)

    # 计算F1分数
    f1 = f1_score(test_labels, predicted_labels)

    # 计算混淆矩阵
    confusion = confusion_matrix(test_labels, predicted_labels)
    tn, fp, fn, tp = confusion.ravel()

    # 计算特异度（Specificity）
    specificity = tn / (tn + fp)

    # 打印指标结果
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall/Sensitivity:', recall)
    print('F1 score:', f1)
    print('Specificity:', specificity)
print('未增强数据结果')
get_metrices(predicted_labels,test_labels)
print('增强数据结果')
get_metrices(predicted_labels_a,test_labels_a)




fpr_original, tpr_original, thresholds_original = roc_curve(test_labels, predicted_labels)
auc_original = auc(fpr_original, tpr_original)

# 计算第二组预测结果的TPR和FPR
fpr_augmented, tpr_augmented, thresholds_augmented = roc_curve(test_labels_a, predicted_labels_a)
auc_augmented = auc(fpr_augmented, tpr_augmented)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr_original, tpr_original, color='darkorange', lw=2, label='Original (AUC = %0.2f)' % auc_original)
plt.plot(fpr_augmented, tpr_augmented, color='green', lw=2, label='Augmented (AUC = %0.2f)' % auc_augmented)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('MCI vs. AD')
plt.legend(loc="lower right")
plt.show()
