import numpy as np
import os

basedir = '/mnt/lls/local_export/3/home/choi574/research_mk/foveated_adv_additional_rebuttal22/imagenet-c/from_greatlake'
dir_alexnet = os.path.join(basedir, 'alexnet')
dir_ffcnn = os.path.join(basedir, 'ffcnn/FFCNN0_val1000')
dir_retina = os.path.join(basedir, 'retina', 'retina1d_res64_b12_s2_bugfixstg3__adv_attack12steps_imagenetC')
dir_crop1 = os.path.join(basedir, 'crop1', 'cropSingle1d_res64_s2r2_bugfixstg3__adv_glimpseNew_12steps_SPSA')
dir_crop2 = os.path.join(basedir, 'crop2', 'cropDouble1d_res64_s2r2_bugfixstg3_glimpseNew__adv_12steps')
dir_s3ta = os.path.join(basedir, 's3ta', 's3TA_lastStepLoss_curriculum_head2_102030_AA__attack12steps')


currs = {}
currs['blur'] = ['defocus_blur',  'glass_blur',  'motion_blur',  'zoom_blur']
currs['digital'] = ['contrast',  'elastic_transform',  'jpeg_compression',  'pixelate']
currs['extra'] = ['gaussian_blur',  'saturate',  'spatter',  'speckle_noise']
currs['noise'] = ['gaussian_noise',  'impulse_noise',  'shot_noise']
currs['weather'] = ['brightness',  'fog',  'frost',  'snow']

def calc_mCe(model_data):
    currs = {}
    currs['blur'] = ['defocus_blur',  'glass_blur',  'motion_blur',  'zoom_blur']
    currs['digital'] = ['contrast',  'elastic_transform',  'jpeg_compression',  'pixelate']
    currs['extra'] = ['gaussian_blur',  'saturate',  'spatter',  'speckle_noise']
    currs['noise'] = ['gaussian_noise',  'impulse_noise',  'shot_noise']
    currs['weather'] = ['brightness',  'fog',  'frost',  'snow']

    model_mCE = []
    for curr_type in currs.keys():
        for curr_type_sub in currs[curr_type]:
            acc_accum = 0.0
            for lev in range(5):
                tag = [curr_type, curr_type_sub, str(lev+1)].join('/')
                acc = model_data.item()[tag][0].item()
                acc_accum = acc_accum + acc
            acc_accum /= 5
            model_mCE.append(acc_accum)

    return np.asarray(model_mCE)





alexnet_t1 = np.load(os.path.join(dir_alexnet, 'records_t1.npy'), allow_pickle=True)
alexnet_t1_mCE = calc_mCe(alexnet_t1)

ffcnn_t1 = np.load(os.path.join(dir_ffcnn, 'records_t1.npy'), allow_pickle=True)
ffcnn_t1_mCE = calc_mCe(ffcnn_t1) / alexnet_t1_mCE
ffcnn_t1_mCE_avg = np.mean(ffcnn_t1_mCE)

retina_t1 = np.load(os.path.join(dir_retina, 'records_t1.npy'), allow_pickle=True)
retina_t1_mCE = calc_mCe(retina_t1) / alexnet_t1_mCE
retina_t1_mCE_avg = np.mean(retina_t1_mCE)

crop1_t1 = np.load(os.path.join(dir_crop1, 'records_t1.npy'), allow_pickle=True)
crop1_t1_mCE = calc_mCe(crop1_t1) / alexnet_t1_mCE
crop1_t1_mCE_avg = np.mean(crop1_t1_mCE)

crop2_t1 = np.load(os.path.join(dir_crop2, 'records_t1.npy'), allow_pickle=True)
crop2_t1_mCE = calc_mCe(crop2_t1) / alexnet_t1_mCE
crop2_t1_mCE_avg = np.mean(crop2_t1_mCE)

s3ta_t1 = np.load(os.path.join(dir_s3ta, 'records_t1.npy'), allow_pickle=True)
s3ta_t1_mCE = calc_mCe(s3ta_t1) / alexnet_t1_mCE
s3ta_t1_mCE_avg = np.mean(s3ta_t1_mCE)



print(retina_t1_mCE_avg, crop1_t1_mCE_avg, crop2_t1_mCE_avg, ffcnn_t1_mCE_avg)




