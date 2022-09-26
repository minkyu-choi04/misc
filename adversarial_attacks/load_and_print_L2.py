import numpy as np

def load_and_print_L2(c_list):
    # Run inside data dir. 
    res = []
    for cval in c_list:
        data = np.load('./perturb_l2_e'+str(int(cval*10000))+'.npy')
        res.append(np.mean(data))
    print(res)


c_list = [100, 30, 10, 3, 1, 0.3, 0.1, 0.03, 0.01]
load_and_print_L2(c_list)
