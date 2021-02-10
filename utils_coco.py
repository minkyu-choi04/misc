import numpy as np

#source_dir = '/mnt/lls/local_export/3/home/choi574/git_libs/misc/coco-labels-2014_2017.txt'
source_dir = '/home/choi574/git_libs/misc/coco-labels-2014_2017.txt'
label_num2str = {}
label_str2num = {}
labels_all = []
cnt = 0
with open(source_dir) as f:
    labels = f.readlines()
    for label in labels:
        label_num2str[str(cnt)] = label[:-1]
        label_str2num[label[:-1]] = cnt
        labels_all = labels_all + [label[:-1]]
        cnt += 1

def lookup_coco_num2str(num):
    ''' input: 
            num: 0~79 integer
        output:
            label: string label corresponding to num
    '''
    return label_num2str[str(num)]

def lookup_coco_str2num(label):
    return label_str2num[label]

#print(lookup_coco_num2str(0))
#print(lookup_coco_str2num('bicycle'))

def visualize_sorted_labels(targets, preds):
    ''' 2021.02.06
    It works for one image sample at a time. 
    inputs: 
            targets: numpy array (80x1), elements are either 0 or 1
            preds: numpy array (80x1), elements are sigmoid(logits) of prediction thus 0<val<1
    outputs:
            sorted_labels_idx: numpy array (80x1), including index of all labels. This is sorted one of 0~79 by the value of 'targets'
            sorted_targets: numpy array (80x1), sorted by its elemets. 
            sorted_preds: numpy array (80x1), sorted by the values of  'targets'. 
            tot_sorted: big array (80,4) with targets, preds, label_idx, _label_all included
    '''
    labels_idx = np.expand_dims(np.array([i for i in range(80)]), 1) # (80,1)
    targets = np.expand_dims(np.squeeze(targets), 1)
    preds = np.expand_dims(np.squeeze(preds), 1)
    _labels_all = np.expand_dims(np.squeeze(np.array(labels_all)), 1)

    #print(labels_idx.shape, targets.shape, preds.shape, _labels_all.shape)
    tot = np.concatenate((targets, preds, labels_idx, _labels_all), 1).tolist() #(80x3)
    tot_sorted = sorted(tot, reverse=True)
    return tot_sorted



    

