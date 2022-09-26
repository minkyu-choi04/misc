import numpy as np

def attack_success_rate_oneStep(tf_clean, tf_adv, attack_step=4):
    '''
    tf_clean: true/false np array, (n_images, n_steps)
    tf_adv: true/false np array, (n_images, n_steps)
    '''
    clean = tf_clean[:, attack_step-1].astype(int).astype(float)
    adv = tf_adv[:, attack_step-1].astype(int).astype(float)
    #print(clean[:10], adv[:10])
    
    n_clean_correct = np.sum(clean)
    n_adv_correct = np.sum(clean * adv)
    n_adv_wrong = n_clean_correct - n_adv_correct

    #print(n_clean_correct, n_adv_correct, n_adv_wrong)
    return n_adv_correct/n_clean_correct

def attack_success_rate_allSteps(tf_clean, tf_adv, attack_step=4):
    '''
    tf_clean: true/false np array, (n_images, n_steps)
    tf_adv: true/false np array, (n_images, n_steps)
    '''
    clean = tf_clean[:, attack_step-1].astype(int).astype(float)
    adv = tf_adv[:, attack_step-1].astype(int).astype(float)
    #print(clean[:10], adv[:10])
    
    n_clean_correct = np.sum(clean)
    n_adv_correct = np.sum(clean * adv)
    n_adv_wrong = n_clean_correct - n_adv_correct

    #print(n_clean_correct, n_adv_correct, n_adv_wrong)
    return n_adv_wrong/n_clean_correct



#tf_clean = np.load('tf_clean_e10.npy')
#tf_adv = np.load('tf_adv_e10.npy')

#attack_success_rate(tf_clean, tf_adv)

#eps_list = [0.0, 0.002, 0.003, 0.005, 0.001, 0.0015, 0.0025, 0.01, 0.0001, 0.0003, 0.0005, 0.05]
#eps_list = [0.0001, 0.0003, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.005, 0.01, 0.03, 0.05]
eps_list = [0.003, 0.005]
asr = []
for eps in eps_list:
    fn_adv = 'tf_adv_e'+str(int(eps*10000))+'.npy'
    fn_clean = 'tf_clean_e'+str(int(eps*10000))+'.npy'
    
    tf_clean = np.load(fn_clean)
    tf_adv = np.load(fn_adv)
    asr.append(attack_success_rate_oneStep(tf_clean, tf_adv))
    #print(eps, fn_adv, eps*10000, int(eps*10000), str(int(eps*10000)))
    
print(asr)
