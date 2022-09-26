import numpy as np

def calculate_ASR_targeted(eps = [0.009, 0.007, 0.005, 0.003, 0.002, 0.001, 0.03, 0.02, 0.01]):
    fn_open_clean = 'tf_clean_e'+str(0)+'.npy'
    data_clean = np.load(fn_open_clean) # 1000x16
    print('clean =  [')
    print(np.mean(data_clean, 0))
    print(']')

    for e in eps:
        fn_open = 'tf_adv_e'+str(int(e*10000))+'.npy'
        data_adv = np.load(fn_open) # 1000x16
        print(np.shape(data_adv))

        clean_adv_TF = data_clean * data_adv
        #print('e'+str(int(e*10000))+' = [', np.mean(data, 0) + ']')
        print('e'+str(int(e*10000))+' = ')
        print(np.sum(clean_adv_TF, 0) / np.sum(data_clean, 0))



def calculate_ASR_untargeted(eps = [0.01, 0.007, 0.005, 0.003, 0.002]):
    fn_open_clean = 'tf_clean_e'+str(0)+'.npy'
    data_clean = np.load(fn_open_clean) # (1 x #samples)
    print('clean acc = ', np.mean(data_clean))

    for e in eps:
        fn_open = 'tf_adv_e'+str(int(e*10000))+'.npy'
        data_adv = np.load(fn_open) # 1 x #samples 
        #print(np.shape(data_adv))

        n_attack_success = 0
        for idx_sample in range(len(data_clean[0])):
            if data_clean[0][idx_sample] == True:
                if data_adv[0][idx_sample] == False:
                    n_attack_success = n_attack_success + 1




        #clean_adv_TF = data_clean * data_adv
        #print('e'+str(int(e*10000))+' = [', np.mean(data, 0) + ']')
        print('e'+str(int(e*10000))+' = ')
        #print(np.sum(clean_adv_TF, 0) / np.sum(data_clean, 0))
        #print(n_attack_success / np.sum(data_clean, 0))
        print(n_attack_success / np.sum(data_clean[0]))



if __name__ == "__main__":
    calculate_ASR_untargeted()
