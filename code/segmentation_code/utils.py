import numpy as np
from collections import deque

import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

def onboard(y, x, shape):
    return (y >= 0) and (y <= shape[0] - 1) and (x >= 0) and (x <= shape[1] - 1)

def fill_gaps(prediction):
    used_mask = np.zeros_like(prediction)
    for y in range(used_mask.shape[0]):
        for x in range(used_mask.shape[1]):
            if prediction[y][x] == 1:
                continue
            elif used_mask[y][x] == 0:
                cur_deq = deque()
                candidates = []
                surrounded = True
                cur_deq.append((y, x))
                candidates.append((y, x))
                while len(cur_deq) > 0:
                    cur_y, cur_x = cur_deq.popleft()
                    used_mask[cur_y][cur_x] = 1
                    if ((cur_y == 0) or (cur_y == used_mask.shape[0] - 1) or (cur_x == 0) or (cur_x == used_mask.shape[1] - 1)):
                        surrounded = False
                    for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]:        
                        if onboard(cur_y + dy, cur_x + dx, used_mask.shape) and \
                        used_mask[cur_y + dy][cur_x + dx] == 0 and \
                        prediction[cur_y + dy][cur_x + dx] == 0:
                            used_mask[cur_y + dy][cur_x + dx] = 1
                            cur_deq.append((cur_y + dy, cur_x + dx))
                            candidates.append((cur_y + dy, cur_x + dx))
                if surrounded:
                    for cur_y, cur_x in candidates:
                        prediction[cur_y][cur_x] = 1
                        
    return prediction 

def calc_iou(prediction, ground_truth):
    n_images = len(prediction)
    intersection, union = 0, 0
    for i in range(n_images):
        intersection += np.logical_and(prediction[i] > 0, ground_truth[i] > 0).astype(np.float32).sum() 
        union += np.logical_or(prediction[i] > 0, ground_truth[i] > 0).astype(np.float32).sum()

    if union == 0:
        union += 0.000001
    return float(intersection) / union

class model_wrapper(pl.LightningModule):
    def __init__(self, base, loss_function=torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.model = base
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.loss_function = loss_function
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_num):
        img = batch['image']
        mask = batch['mask']
        
        probs = self(img).squeeze(1)
        loss = self.loss_function(probs, mask.long())

        self.training_step_outputs.append({'loss': loss.detach().cpu()})
        return loss
        
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 
                                                                  mode='min', 
                                                                  factor=0.2, 
                                                                  patience=6, 
                                                                  verbose=True)        
        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss"
        } 
        
        return [opt], [lr_dict]
    
    def validation_step(self, batch, batch_idx):
        img = batch['image']
        mask = batch['mask']
        
        probs = self(img).squeeze(1)
        loss = self.loss_function(probs, mask.long())

        prediction = torch.argmax(probs, dim=1).cpu().numpy().astype(np.float32)
        iou = calc_iou(prediction, mask.cpu().numpy())
        
        self.validation_step_outputs.append({'loss': loss.cpu(), 'iou': iou})
        return {'loss': loss, 'iou': iou}
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        avg_iou = np.mean([x['iou'] for x in self.validation_step_outputs])
        self.log("val_loss", avg_loss)
        self.log("val_iou", avg_iou)
        print(f"Epoch {self.trainer.current_epoch},Val_loss: {round(float(avg_loss), 3)}, Val_iou: {round(float(avg_iou), 3)}")
        self.validation_step_outputs.clear()
        
    
    def on_training_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
        self.log("train_loss", avg_loss)
        print(f"Epoch {self.trainer.current_epoch}, Train_loss: {round(float(avg_loss), 3)}")
        self.training_step_outputs.clear()

def to_one_hot(tensor, nClasses):
    n,h,w = tensor.size()
    one_hot = torch.zeros(n,nClasses,h,w).scatter_(1,tensor.view(n,1,h,w).cpu(),1)
    return one_hot

class SoftIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super().__init__()
        self.classes = n_classes

    def forward(self, inputs, target):
        target_oneHot = to_one_hot(target, nClasses=self.classes).cuda()
        N = inputs.size()[0]
        
        inputs = F.softmax(inputs,dim=1)
        
        inter = inputs * target_oneHot
        inter = inter.view(N,self.classes,-1).sum(2)
        
        union= inputs + target_oneHot - (inputs*target_oneHot)
        union = union.view(N,self.classes,-1).sum(2)
        
        loss = inter/union
        
        return -loss.mean()