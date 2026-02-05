import torch
import torch.nn.functional as F
import torch.nn as nn

my_weight = torch.FloatTensor([0.05, 1.0]).cuda()
criterion = nn.NLLLoss(ignore_index=255, weight=my_weight, reduction="none")

def weit_loss(pred, mask, weit, smooth=1e-6): 

    weit = weit * 2 + 1
    mask_one_hot = F.one_hot(mask.long(), num_classes=2).permute(0, 3, 1, 2).float()
    wbce = criterion(torch.log(torch.clamp(pred, torch.finfo(torch.float32).eps,
                                                         1 - torch.finfo(torch.float32).eps)), mask)
    wbce = (weit * wbce).mean()
    
    # IOU
    # pred_sig = torch.softmax(pred)
    inter = (pred * mask_one_hot * weit).sum(dim=(2, 3))
    union = (pred * weit).sum(dim=(2, 3)) + (mask_one_hot * weit).sum(dim=(2, 3)) - inter
    wiou = 1 - ((inter+smooth)/(union+smooth)).mean(dim=1)
    
    if wiou.mean() < 0:
        print("Error", wiou.mean())
        return wbce
        
    loss = wbce + wiou.mean()
    return loss