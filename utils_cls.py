import torch

def pred_label(pred_logit):
    ''' return most probable labels from prediction from the network
    pred_logit: (b, c)
    labels_pred: (b,)
    '''
    _, labels_pred = torch.max(torch.squeeze(pred_logit.data), 1)
    return labels_pred

def tf_label(pred_logit, labels):
    '''return true or false by comparing predicted labels and GT labels'''
    labels_pred = pred_label(pred_logit)
    tf = torch.squeeze(labels_pred.eq(torch.squeeze(labels).data.view_as(torch.squeeze(labels_pred))))
    return tf
