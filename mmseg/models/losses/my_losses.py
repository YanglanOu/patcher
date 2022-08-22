import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss



@LOSSES.register_module()
class StructurelLoss(nn.Module):

    def __init__(self,
                 smoothness=1.0, **kwargs):
        super().__init__()
        self.smoothness = smoothness


    def forward(self,
                pred, mask,
                **kwargs):
        """Forward function."""

        # assert torch.all(mask <= 1)
        mask = mask.clone()
        mask[mask == 255] = 0
        mask_f = mask.clone().float().unsqueeze(1)
        weit = (1 + 5*torch.abs(F.avg_pool2d(mask_f, kernel_size=31, stride=1, padding=15) - mask_f)).squeeze(1)
        wbce = F.cross_entropy(pred, mask, reduction='none', ignore_index=255)
        wbce = (weit*wbce).sum(dim=(1, 2)) / weit.sum(dim=(1, 2))

        prob = torch.softmax(pred, dim=1)
        pred = prob[:, 1]
        inter = ((pred * mask) * weit).sum(dim=(1, 2))
        cardinality = ((pred + mask) * weit).sum(dim=(1, 2))
        union = cardinality - inter
        wiou = 1 - (inter + self.smoothness)/(union + self.smoothness)
        
        return (wbce + wiou).mean()


# def structure_loss(pred, mask):
    
#     weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
#     wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
#     wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

#     pred = torch.sigmoid(pred)
#     inter = ((pred * mask)*weit).sum(dim=(2, 3))
#     cardinality = ((pred + mask)*weit).sum(dim=(2, 3))
#     union = cardinality - inter
#     wiou = -(inter + 1)/(union + 1)
    
#     return (wbce + wiou).mean()