import numpy as np
from glob import glob
import os
import csv
import math
from shutil import copy
import matplotlib.pyplot as plt
import ants
from scipy import stats
from scipy.stats import ttest_ind, multivariate_normal, chi2_contingency


def get_files(path):
    file_list = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_list.append(os.path.join(dirpath, filename))
    return file_list


path = "/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Dataset/ADNI"
file_list = get_files(path)


subject_dict = {}
# 打开csv文件并读取数据
with open('/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Dataset/ADNI.csv', 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)

    # 遍历每一行数据
    for row in csv_reader:
        # 获取当前行的Subject值
        subject = row['Subject']

        # 将当前行数据存储到相应的Subject值对应的列表中
        if subject in subject_dict:
            subject_dict[subject].append(row)
        else:
            subject_dict[subject] = [row]

# 显示每一行
for subject, rows in subject_dict.items():
    print(f"Subject {subject}:")
    for row in rows:
        print(row)


datapath_dirs = sorted(glob('/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Dataset/ADNI/*'))

new_dataset = []
for datapath_dir in datapath_dirs:
    subject_id = datapath_dir.split('/')[-1] # 获取Subject_id
    subjects = subject_dict[subject_id] # 获得这个人的不同检测
    # 根据年龄取中值 获得中值对应的检测图像
    age_id_dict = {d['Age']: d['Image Data ID'] for d in subjects}
    age_list = sorted([int(age) for age in age_id_dict.keys()])
    age = age_list[math.floor(len(age_list)/2)]
    image_id = age_id_dict[str(age)]

    image_files = get_files(datapath_dir)
    for image_file in image_files:
        image_file_id = image_file.split('_')[-1].split('.')[0]
        if image_file_id == image_id:
            new_dataset.append(image_file)
print(len(new_dataset))


with open('/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Dataset/ADNI.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    # 创建一个字典，将Image Data ID和相应的值对应起来
    data_dict = {row['Image Data ID']: row for row in reader}

ADNI_501 = './Dataset/ADNI_501/'
_ = os.mkdir(ADNI_501) if not os.path.exists(ADNI_501) else 0

CN_Male_nums = 0
CN_Female_nums = 0
MCI_Male_nums = 0
MCI_Female_nums = 0
CN_Age = []
MCI_Age = []
for data_path in new_dataset:
    image_id = data_path.split('_')[-1].split('.')[0]
    values = data_dict[image_id] # 获取Age这一列的值
    group = values['Group']
    sex = values['Sex']
    age = values['Age']
    if group == 'CN':
        CN_Age.append(int(age))
        if sex == 'M':
            CN_Male_nums+=1
            sex = 0
        else:
            CN_Female_nums += 1
            sex = 1
        group = 0
    else:
        MCI_Age.append(int(age))
        if sex == 'M':
            MCI_Male_nums+=1
            sex = 0
        else:
            MCI_Female_nums += 1
            sex = 1
        group = 1

    new_data_path = ADNI_501 + image_id + '_' + str(group) + '_' + age + '_' + str(sex) + '.nii'
    print(new_data_path)
    copy(data_path,new_data_path)
print(CN_Male_nums,CN_Female_nums)
print(MCI_Male_nums,MCI_Female_nums)
print(sorted(CN_Age),sorted(MCI_Age))
# print(sorted(Ages))


t, p =ttest_ind(sorted(CN_Age),sorted(MCI_Age))

# 年龄列表
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
# 绘制直方图
ages = CN_Age
ax[0].hist(ages, bins=range(min(ages), max(ages) + 2, 2))
ages = MCI_Age
ax[1].hist(ages, bins=range(min(ages), max(ages) + 2, 2))
# 添加标签和标题
# ax[0].xlabel('Age')
# ax[0].ylabel('Frequency')
# ax[0].title('Age Distribution')

# 显示图形
plt.show()

data = [CN_Male_nums,CN_Female_nums]

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
labels = ['Male', 'Female']
# 设置颜色
colors = ['#ff9999','#66b3ff']
# 创建饼图
ax[0].pie(data, labels=labels, colors=colors, autopct='%1.1f%%')
ax[0].set_title('CN')
data = [MCI_Male_nums,MCI_Female_nums]
ax[1].pie(data, labels=labels, colors=colors, autopct='%1.1f%%')
ax[1].set_title('MCI')
data = [CN_Male_nums+MCI_Male_nums,CN_Female_nums+MCI_Female_nums]
ax[2].pie(data, labels=labels, colors=colors, autopct='%1.1f%%')
ax[2].set_title('total')
# 显示图形
plt.show()

data = glob('/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Dataset/ADNI_501/*')

spm_dataset_dir = './Dataset/spm_data/'
_ = os.mkdir(spm_dataset_dir) if not os.path.exists(spm_dataset_dir) else 0
spm_data = sorted(glob('./Dataset/mri/*'))
mci_2mm = ants.image_read('/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Template/MCI152_T1_2mm_brain.nii.gz')
for i in range(len(spm_data)):
    iden = spm_data[i].split('/')[-1][0:4]
    # print(iden)
    if iden == 'mwp1':
        d = ants.image_read(spm_data[i])
        d = ants.registration(fixed=mci_2mm,moving=d)['warpedmovout']
        new_data_path = spm_dataset_dir+spm_data[i].split('/')[-1]
        print(new_data_path)
        ants.image_write(d,new_data_path)


fsl_dataset_dir = './Dataset/fsl_data/'
_ = os.mkdir(fsl_dataset_dir) if not os.path.exists(fsl_dataset_dir) else 0
fsl_data = sorted(glob('./Dataset/ADNI_fsl/Seg/*'))
print(len(fsl_data))
for i in range(len(fsl_data)):
    iden = fsl_data[i].split('/')[-1].split('_')[4]
    if iden == 'GM':
        new_data_path = fsl_dataset_dir + fsl_data[i].split('/')[-1]
        # print(new_data_path)
        copy(spm_data[i], new_data_path)


d = ants.image_read('/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Dataset/spm_data/mwp1I100759_1_79_0.nii')
mci_2mm = ants.image_read('/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Template/MCI152_T1_2mm_brain.nii.gz')
print(d.shape)
ants.plot(d)


spm_data = sorted(glob('./Dataset/spm_data/*'))
CN_data = []
MCI_data = []
for i in range(len(spm_data)):
    data = ants.image_read(spm_data[i]).numpy()
    print(np.max(data),np.min(data))
    if spm_data[i].split('/')[-1].split('_')[1] == '0':
        print('CN')
        CN_data.append(data)
    else:
        print('MCI')
        MCI_data.append(data)
CN_data = np.array(CN_data)
MCI_data = np.array(MCI_data)
print(CN_data.shape)
print(MCI_data.shape)
np.save('./Dataset/spm_CN.npy',CN_data)
np.save('./Dataset/spm_MCI.npy',MCI_data)


fsl_data = sorted(glob('./Dataset/fsl_data/*'))
CN_data = []
MCI_data = []
for i in range(len(fsl_data)):
    data = ants.image_read(fsl_data[i]).numpy()
    if fsl_data[i].split('/')[-1].split('_')[1] == '0':
        print('CN')
        CN_data.append(data)
    else:
        print('MCI')
        MCI_data.append(data)
CN_data = np.array(CN_data)
MCI_data = np.array(MCI_data)
print(CN_data.shape)
print(MCI_data.shape)
np.save('./Dataset/fsl_CN.npy', CN_data)
np.save('./Dataset/fsl_MCI.npy', MCI_data)


# AD预处理
AD_path = "/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Dataset/AD/ADNI"
file_list = get_files(AD_path)

subject_dict = {}
# 打开csv文件并读取数据
with open('/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Dataset/WYQADcsv.csv', 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)

    # 遍历每一行数据
    for row in csv_reader:
        # 获取当前行的Subject值
        subject = row['Subject']

        # 将当前行数据存储到相应的Subject值对应的列表中
        if subject in subject_dict:
            subject_dict[subject].append(row)
        else:
            subject_dict[subject] = [row]

for subject, rows in subject_dict.items():
    print(f"Subject {subject}:")
    for row in rows:
        print(row)

datapath_dirs = sorted(glob('/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Dataset/AD/ADNI/*'))

new_dataset = []
for datapath_dir in datapath_dirs:
    subject_id = datapath_dir.split('/')[-1] # 获取Subject_id
    subjects = subject_dict[subject_id] # 获得这个人的不同检测
    # 根据年龄取中值 获得中值对应的检测图像
    age_id_dict = {d['Age']: d['Image Data ID'] for d in subjects}
    age_list = sorted([int(age) for age in age_id_dict.keys()])
    age = age_list[math.floor(len(age_list)/2)]
    image_id = age_id_dict[str(age)]

    image_files = get_files(datapath_dir)
    for image_file in image_files:
        image_file_id = image_file.split('_')[-1].split('.')[0]
        if image_file_id == image_id:
            new_dataset.append(image_file)
print(len(new_dataset))

with open('/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Dataset/WYQADcsv.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    # 创建一个字典，将Image Data ID和相应的值对应起来
    data_dict = {row['Image Data ID']: row for row in reader}

AD_path = './Dataset/AD_195/'
_ = os.mkdir(AD_path) if not os.path.exists(AD_path) else 0

AD_Male_nums = 0
AD_Female_nums = 0
AD_Age = []
for data_path in new_dataset:
    image_id = data_path.split('_')[-1].split('.')[0]
    values = data_dict[image_id] # 获取Age这一列的值
    group = values['Group']
    sex = values['Sex']
    age = values['Age']
    if group == 'AD':
        AD_Age.append(int(age))
        if sex == 'M':
            AD_Male_nums+=1
            sex = 0
        else:
            AD_Female_nums += 1
            sex = 1
        group = 2

    new_data_path = AD_path + image_id + '_' + str(group) + '_' + age + '_' + str(sex) + '.nii'
    print(new_data_path)
    copy(data_path,new_data_path)
print(AD_Male_nums,AD_Female_nums)
# print(MCI_Male_nums,MCI_Female_nums)
print(sorted(AD_Age))


spm_dataset_dir = './Dataset/spm_ADdata/'
_ = os.mkdir(spm_dataset_dir) if not os.path.exists(spm_dataset_dir) else 0
spm_data = sorted(glob('./Dataset/AD/mri/*'))
# print(spm_data)
mci_2mm = ants.image_read('/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Template/MCI152_T1_2mm_brain.nii.gz')
for i in range(len(spm_data)):
    iden = spm_data[i].split('/')[-1][0:4]
    # print(iden)
    if iden == 'mwp1':
        d = ants.image_read(spm_data[i])
        d = ants.registration(fixed=mci_2mm,moving=d)['warpedmovout']
        new_data_path = spm_dataset_dir+spm_data[i].split('/')[-1]
        print(new_data_path)
        ants.image_write(d,new_data_path)


spm_data = sorted(glob('./Dataset/spm_ADdata/*'))
print(len(spm_data))
AD_data = []
for i in range(len(spm_data)):
    data = ants.image_read(spm_data[i]).numpy()
    print(np.max(data),np.min(data))
    AD_data.append(data)
AD_data = np.array(AD_data)
print(AD_data.shape)
np.save('./Dataset/spm_AD.npy',AD_data)

cn_dataset_dir = './Dataset/spm_CNdata/'
mci_dataset_dir = './Dataset/spm_MCIdata/'
_ = os.mkdir(cn_dataset_dir) if not os.path.exists(cn_dataset_dir) else 0
_ = os.mkdir(mci_dataset_dir) if not os.path.exists(mci_dataset_dir) else 0
spm_data = sorted(glob('/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Dataset/spm_data/*'))
for data in spm_data:
    if data.split('/')[-1].split('_')[1] == '0':
        copy(data,cn_dataset_dir+data.split('/')[-1])
    else:
        copy(data,mci_dataset_dir+data.split('/')[-1])

data = sorted(glob('/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Dataset/ADNI_501/*'))
CN_sex = []
CN_age = []
for d in data:
    if d.split('_')[1] == '0':
        CN_sex.append(int(d.split('_')[3]))
        CN_age.append(int(d.split('_')[2]))

MCI_sex = []
MCI_age = []
for d in data:
    if d.split('_')[1] == '1':
        MCI_sex.append(int(d.split('_')[3]))
        MCI_age.append(int(d.split('_')[2]))

from glob import glob
data = sorted(glob('/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Dataset/AD_195/*'))
for i in range(len(data)):
    data[i] = data[i].split('/')[-1].split('.')[0]
AD_sex = []
AD_age = []
for d in data:
    if d.split('_')[1] == '2':
        AD_sex.append(int(d.split('_')[3]))
        AD_age.append(int(d.split('_')[2]))


def F_age(data1, data2,data3):
    # 执行F检验
    f_statistic, p_value = stats.f_oneway(data1, data2,data3)
    # 打印结果
    print("F统计量：", f_statistic)
    print("p值：", p_value)



import numpy as np
CN_MMSE = ['30', '29', '29', '29', '30', '29', '30', '29', '30', '29', '30', '28', '30', '30', '29', '29', '29', '28', '29', '30', '30', '29', '28', '30', '30', '29', '29', '30', '29', '29', '27', '30', '29', '30', '30', '30', '30', '30', '30', '29', '30', '30', '30', '30', '30', '26', '30', '29', '28', '30', '29', '29', '28', '29', '30', '29', '28', '30', '30', '29', '30', '30', '30', '28', '29', '29', '30', '29', '29', '30', '29', '29', '30', '30', '30', '30', '30', '26', '29', '27', '30', '30', '30', '29', '29', '28', '30', '28', '29', '29', '30', '27', '29', '30', '28', '29', '28', '28', '29', '30', '30', '29', '30', '30', '26', '30', '28', '29', '30', '28', '29', '30', '30', '30', '26', '30', '29', '30', '29', '30', '29', '30', '28', '30', '30', '29', '29', '29', '29', '28', '26', '29', '30', '29', '29', '28', '28', '30', '30', '29', '29', '29', '26', '29', '30', '29', '30', '29', '30', '30', '30', '28', '30', '30', '28', '29', '30', '28', '26', '29', '30', '30', '29', '30', '30', '29', '29', '29', '27', '29', '30', '30', '29', '30', '30', '24', '29', '30', '29', '30', '29', '28', '28', '30', '30', '27', '28', '28', '30', '30', '30', '27', '30', '29', '28', '30', '30', '30', '30', '30', '30', '30', '25', '29', '30', '30', '28', '30', '29', '29', '30', '30', '30', '29', '30', '30', '29', '30', '30', '30', '29', '29', '30', '30', '26', '30', '28', '30', '27', '30', '29', '27', '30', '30', '29', '29', '29', '29', '29', '26', '29', '30', '30', '29', '30', '28', '27', '28', '30', '30', '25', '28', '29', '30', '30', '29', '29', '29', '30', '30', '30', '30', '30', '29', '28', '30', '27', '28', '28', '29', '29', '28', '29', '29', '30', '28', '30', '29', '29', '30', '29', '29', '30', '30', '26']
CN_MMSE = np.array(CN_MMSE).astype(np.float64)
MCI_MMSE = ['27', '27', '28', '29', '28', '24', '26', '30', '26', '26', '26', '29', '25', '26', '27', '26', '28', '28', '27', '26', '29', '28', '25', '30', '25', '26', '24', '28', '29', '29', '26', '26', '29', '24', '26', '26', '30', '29', '28', '27', '27', '26', '26', '27', '25', '29', '29', '25', '27', '28', '26', '26', '28', '27', '28', '29', '26', '30', '28', '27', '27', '30', '29', '25', '26', '29', '25', '27', '26', '24', '25', '30', '27', '28', '27', '24', '28', '24', '29', '27', '29', '24', '27', '28', '28', '27', '28', '24', '26', '27', '27', '28', '30', '27', '28', '29', '29', '25', '25', '24', '26', '30', '29', '27', '30', '28', '25', '27', '30', '29', '26', '25', '25', '27', '27', '24', '28', '27', '27', '29', '25', '25', '27', '24', '29', '28', '26', '28', '27', '27', '28', '24', '26', '29', '24', '27', '29', '27', '29', '24', '29', '27', '27', '28', '26', '25', '30', '29', '27', '27', '27', '26', '25', '25', '29', '26', '26', '26', '30', '28', '26', '29', '28', '24', '27', '26', '29', '25', '27', '25', '28', '28', '26', '27', '26', '27', '29', '28', '25', '23', '24', '28', '29', '30', '28', '25', '28', '25', '29', '29', '26', '26', '28', '29', '29', '25', '27', '27', '29', '27', '24', '25', '27', '24', '24', '28', '28', '24', '24', '28', '27', '26', '30', '25', '26', '30']
MCI_MMSE = np.array(MCI_MMSE).astype(np.float64)
AD_MMSE = ['21', '24', '21', '24', '24', '20', '23', '20', '20', '26', '20', '22', '21', '22', '26', '26', '25', '20', '25', '26', '23', '22', '24', '22', '18', '23', '21', '22', '25', '26', '26', '20', '23', '23', '20', '26', '21', '19', '23', '22', '21', '21', '23', '22', '22', '20', '21', '23', '23', '22', '25', '24', '22', '22', '24', '22', '26', '26', '21', '20', '22', '24', '24', '21', '24', '26', '19', '26', '22', '24', '21', '24', '22', '25', '23', '22', '25', '20', '21', '21', '20', '20', '23', '22', '26', '26', '23', '25', '20', '25', '21', '25', '21', '24', '25', '25', '25', '21', '26', '26', '25', '26', '22', '23', '25', '21', '23', '26', '24', '26', '22', '26', '26', '25', '26', '24', '20', '19', '21', '24', '22', '26', '24', '26', '22', '20', '20', '26', '26', '21', '23', '25', '21', '23', '20', '21', '23', '25', '22', '19', '21', '25', '25', '26', '23', '25', '24', '26', '21', '26', '24', '23', '20', '21', '26', '21', '22', '23', '24', '20', '22', '25', '20', '23', '25', '23', '21', '23', '20', '21', '22', '24', '26', '21', '26', '25', '21', '26', '25', '26', '23', '22', '20', '22', '26', '21', '21', '23', '23', '26', '24', '25', '20', '23', '26']
AD_MMSE = np.array(AD_MMSE).astype(np.float64)

F_age(CN_MMSE, MCI_MMSE, AD_MMSE)



mean = np.array([1, 2])

# 协方差矩阵
covariance_matrix = np.array([[2, 0.5],
                             [0.5, 1]])

# 生成网格点用于绘制等高线图
x, y = np.mgrid[-1:4:.01, -1:5:.01]
pos = np.dstack((x, y))

# 计算多维高斯分布的概率密度函数
pdf = multivariate_normal.pdf(pos, mean=mean, cov=covariance_matrix)

# 绘制等高线图
plt.figure(figsize=(8, 6))
plt.contourf(x, y, pdf, cmap='Blues')
plt.colorbar(label='Probability Density')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Multivariate Gaussian Distribution')
plt.grid(True)
plt.show()

data = np.array([[135, 150], [132, 84], [106, 89]])

# 进行卡方检验
chi2, p_value, dof, expected = chi2_contingency(data)


