import torch
import torch.nn as nn

class EOT(nn.Module):
    '''20210628
    Wrapper for EOT. 
    Args:
        model: classification model returning all information (isReturnAllStep==True). 
    '''
    def __init__(self, model, n_repeats, isAvgSteps):
        super().__init__()
        self.model = model
        self.n_repeats = n_repeats
        self.isAvgSteps = isAvgSteps

    def forward(self, img):
        acc_r = 0.0
        for rp in range(self.n_repeats):
            return_dict = self.model(img)
            # return_dict: (step, batch, c)
            pred_curr = torch.stack(return_dict['pred'])

            acc_r = acc_r + pred_curr
        acc_r = acc_r / self.n_repeats
        # acc_r: (step, batch, c)

        if self.isAvgSteps:
            return torch.mean(acc_r, 0)
        else:
            return torch.squeeze(acc_r[-1, :, :])
    
    def inference(self, img, n_steps):
        rd = self.model(img, n_steps)
        return rd

