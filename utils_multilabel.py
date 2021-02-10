import torch

def calculate_metrics(output, target, m_dict):
    '''2021.02.06
    https://github.com/allenai/elastic/blob/master/multilabel_classify.py
    '''
    pred = output.data.gt(0.0).long()

    m_dict['tp'] += (pred + target).eq(2).sum(dim=0)
    m_dict['fp'] += (pred - target).eq(1).sum(dim=0)
    m_dict['fn'] += (pred - target).eq(-1).sum(dim=0)
    m_dict['tn'] += (pred + target).eq(0).sum(dim=0)

    this_tp = (pred + target).eq(2).sum()
    this_fp = (pred - target).eq(1).sum()
    this_fn = (pred - target).eq(-1).sum()
    this_tn = (pred + target).eq(0).sum()
    
    m_dict['this_acc'] = (this_tp + this_tn).float() / (this_tp + this_tn + this_fp + this_fn).float()
    m_dict['this_prec'] = this_tp.float() / (this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
    m_dict['this_rec'] = this_tp.float() / (this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0

    m_dict['p_c'] = [float(m_dict['tp'][i].float() / (m_dict['tp'][i] + m_dict['fp'][i]).float()) * 100.0 if m_dict['tp'][i] > 0 else 0.0 for i in range(len(m_dict['tp']))]
    m_dict['r_c'] = [float(m_dict['tp'][i].float() / (m_dict['tp'][i] + m_dict['fn'][i]).float()) * 100.0 if m_dict['tp'][i] > 0 else 0.0 for i in range(len(m_dict['tp']))]
    m_dict['f_c'] = [2 * m_dict['p_c'][i] * m_dict['r_c'][i] / (m_dict['p_c'][i] + m_dict['r_c'][i]) if m_dict['tp'][i] > 0 else 0.0 for i in range(len(m_dict['tp']))]

    m_dict['mean_p_c'] = sum(m_dict['p_c']) / len(m_dict['p_c'])
    m_dict['mean_r_c'] = sum(m_dict['r_c']) / len(m_dict['r_c'])
    m_dict['mean_f_c'] = sum(m_dict['f_c']) / len(m_dict['f_c'])

    m_dict['p_o'] = m_dict['tp'].sum().float() / (m_dict['tp'] + m_dict['fp']).sum().float() * 100.0
    m_dict['r_o'] = m_dict['tp'].sum().float() / (m_dict['tp'] + m_dict['fn']).sum().float() * 100.0
    m_dict['f_o'] = 2 * m_dict['p_o'] * m_dict['r_o'] / (m_dict['p_o'] + m_dict['r_o'])

    return m_dict


class AverageMeter_multilabel(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.cnt = 0

    def update(self, output, target):
        ''' 2021.02.06
        Calculate metrics for multi-label classification
        Inputs:
                output: (batch_size, n_classes), tensor, prediction output without sigmoid
                target: (batch_size, n_classes), tensor, elements are either 1 (presense) or 0 (non-presense)
        '''
        # define threshold probability as 0.5 (sigmoid(0.0) = 0.5)
        pred = output.data.gt(0.0).long() # (b, c)

        self.tp += (pred + target).eq(2).sum(dim=0) 
        self.fp += (pred - target).eq(1).sum(dim=0)
        self.fn += (pred - target).eq(-1).sum(dim=0)
        self.tn += (pred + target).eq(0).sum(dim=0)
        self.cnt += target.size(0)

        this_tp = (pred + target).eq(2).sum()
        this_fp = (pred - target).eq(1).sum()
        this_fn = (pred - target).eq(-1).sum()
        this_tn = (pred + target).eq(0).sum()
        this_acc = (this_tp + this_tn).float() / (this_tp + this_tn + this_fp + this_fn).float()

        self.this_prec = this_tp.float() / (this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
        self.this_rec = this_tp.float() / (this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0

        # categorical precision, recall, f1 score
        p_c = [float(self.tp[i].float() / (self.tp[i] + self.fp[i]).float()) * 100.0 if self.tp[i] > 0 else 0.0 for i in range(len(self.tp))]
        r_c = [float(self.tp[i].float() / (self.tp[i] + self.fn[i]).float()) * 100.0 if self.tp[i] > 0 else 0.0 for i in range(len(self.tp))]
        f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if self.tp[i] > 0 else 0.0 for i in range(len(self.tp))]

        self.mean_p_c = sum(p_c) / len(p_c)
        self.mean_r_c = sum(r_c) / len(r_c)
        self.mean_f_c = sum(f_c) / len(f_c)

        self.p_o = self.tp.sum().float() / (self.tp + self.fp).sum().float() * 100.0 # overall precision
        self.r_o = self.tp.sum().float() / (self.tp + self.fn).sum().float() * 100.0 # overall recall
        self.f_o = 2 * self.p_o * self.r_o / (self.p_o + self.r_o) # f1 score


