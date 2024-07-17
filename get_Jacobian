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
