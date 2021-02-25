import numpy as np



def print_every_n_lines(n, fn_r='print_reward_test'):
    cnt = 1
    fn_w = fn_r + '_lastStep'
    f_w = open(fn_w, 'a')
    with open(fn_r) as f_r:
        lines = f_r.readlines()
        for line in lines:
            if cnt % n == 0:
                f_w.write(line)
            cnt += 1
    f_w.close()
    f_r.close()


n=4
fn_r='print_reward_test'
cnt = 1
fn_w = fn_r + '_lastStep'
f_w = open(fn_w, 'a')
with open(fn_r) as f_r:
    lines = f_r.readlines()
    for line in lines:
        if cnt % n == 0:
            f_w.write(line)
        cnt += 1

f_w.close()
f_r.close()



