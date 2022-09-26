import os
import sys
sys.path.append('/mnt/lls/local_export/3/home/choi574/git_libs/misc/')
sys.path.append('/home/choi574/research_mk/git_libs_onlyForLibi2/misc/')
import misc
import numpy as np

def load_and_print_L2(c_list):
    # Run inside data dir. 
    res = []
    for cval in c_list:
        data = np.load('./perturb_l2_e'+str(int(cval*10000))+'.npy')
        res.append(np.mean(data))
    print(res)
        




def write_scores_in_a_row(fn, score_list):
    fd = open(fn, 'a')
    for i in range(len(score_list)):
        fd.write('{:.4f},  '.format(score_list[i]))
    fd.write('\n')
    fd.close()

def make_dir(fn):
    if not os.path.exists(fn):
        os.makedirs(fn)

def save_AA_imgs(imgsavedir, eps, gt_labels, cnt, img_aa):
    '''
    imgsavedir: dir to save images
    eps: float
    gt_labels: int
    cnt: numpy array (100,) counting GT labels that appeared until now. 
    img_aa: tensor,  (1, 3, H, W), the range of the image pixels must be normalized -2.5~2.5. 
    '''


    '''
    Better to use this function with dataloader shuffle_test=False
    '''

    make_dir(imgsavedir)
    dir_eps = os.path.join(imgsavedir, 'e'+str(int(eps*10000)))
    make_dir(dir_eps)
    dir_eps_label = os.path.join(dir_eps, 'c'+str(gt_labels))
    make_dir(dir_eps_label)

    fn = 'i'+str(int(cnt[gt_labels]))+'.jpg'
    #dir_eps_label_img = os.path.join(dir_eps_label, 'i'+fn)
    n_plots=1
    misc.plot_one_sample_from_images(img_aa[:n_plots],
            dir_eps_label, fn)


def save_AA_imgs_range01(imgsavedir, eps, gt_labels, cnt, img_aa):
    '''
    imgsavedir: dir to save images
    eps: float
    gt_labels: int
    cnt: numpy array (100,) counting GT labels that appeared until now. 
    img_aa: tensor,  (1, 3, H, W), the range of the image pixels must be 0~1. 
    '''


    '''
    Better to use this function with dataloader shuffle_test=False
    '''

    make_dir(imgsavedir)
    dir_eps = os.path.join(imgsavedir, 'e'+str(int(eps*10000)))
    make_dir(dir_eps)
    dir_eps_label = os.path.join(dir_eps, 'c'+str(gt_labels))
    make_dir(dir_eps_label)

    fn = 'i'+str(int(cnt[gt_labels]))+'.jpg'
    #dir_eps_label_img = os.path.join(dir_eps_label, 'i'+fn)
    n_plots=1
    misc.plot_one_sample_from_images(img_aa[:n_plots],
            dir_eps_label, fn, isRange01=True)

    
def save_AA_imgs_range01_lookup(imgsavedir, eps, gt_labels, cnt, img_aa, class_name):
    '''
    imgsavedir: dir to save images
    eps: float
    gt_labels: int
    cnt: numpy array (100,) counting GT labels that appeared until now. 
    img_aa: tensor,  (1, 3, H, W), the range of the image pixels must be 0~1. 
    '''


    '''
    Better to use this function with dataloader shuffle_test=False
    '''

    make_dir(imgsavedir)
    dir_eps = os.path.join(imgsavedir, 'e'+str(int(eps*10000)))
    make_dir(dir_eps)
    dir_eps_label = os.path.join(dir_eps, class_name)
    make_dir(dir_eps_label)

    fn = 'i'+str(int(cnt[gt_labels]))+'.jpg'
    #dir_eps_label_img = os.path.join(dir_eps_label, 'i'+fn)
    n_plots=1
    misc.plot_one_sample_from_images(img_aa[:n_plots],
            dir_eps_label, fn, isRange01=True)

