import os
from multiprocessing import Pool as Pool

Standard_image_T1w = '/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Template/MNI152_T1_2mm_brain.nii.gz'
Standard_t1w = '/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Template/MNI152_T1_2mm.nii.gz'
Standard_mask = '/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Template/MNI152_T1_2mm_brain_mask.nii.gz'


def T1w_prep(input_dir, output_dir):
    os.system('mkdir ' + output_dir + 'tmp/')
    os.system('mkdir ' + output_dir + 'tmp/T1/')
    os.system('mkdir ' + output_dir + 'Seg')
    T1_files = os.listdir(input_dir)
    for i in range(len(T1_files)):
        print(i,'%',len(T1_files))
        T1_name = T1_files[i]
        T1_file = os.path.join(input_dir, T1_files[i])
        print(T1_name)
        os.system('robustfov -i ' + T1_file + ' -r ' + output_dir + 'tmp/T1/crop_' + T1_name)
        os.system(
            'antsBrainExtraction.sh -d 3 -a ' + output_dir + 'tmp/T1/crop_' + T1_name + '.gz' + ' -e ' + Standard_t1w + ' -m ' + Standard_mask + ' -o ' + output_dir + 'tmp/T1/BE')

        os.system(
            'antsRegistrationSyN.sh -d 3 -f ' + Standard_image_T1w + ' -m ' + output_dir + 'tmp/T1/BEBrainExtractionBrain.nii.gz  -o ' \
            + output_dir + 'tmp/T1/rega2t -n 20')

        os.system(
            'fast -S 1 -t 1 -o ' + output_dir + 'Seg/seg -g -n 3 -b -I 10 ' + output_dir + 'tmp/T1/rega2tWarped.nii.gz')

        os.system('mv ' + output_dir + 'Seg/seg_seg_2.nii.gz ' + output_dir + 'Seg/' + T1_name.split('.')[0] + '_WM_mask.nii.gz')
        os.system(
            'mv ' + output_dir + 'Seg/seg_seg_1.nii.gz ' + output_dir + 'Seg/' + T1_name.split('.')[0] + '_GM_mask.nii.gz')  # mask 灰质的一般不需要


T1w_prep('/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Dataset/ADNI_501/',
         '/root/commonfiles/wangyuqi/Sex/ADNI_CVAE/Dataset/ADNI_fsl/')
