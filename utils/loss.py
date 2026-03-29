import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from .tal import TaskAlignedAssigner
from .box_utils import make_anchors, dist2bbox, bbox2dist, bbox_iou
        
class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
    
        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl
    # dis loss 
    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)


class VarientFocalLoss(nn.Module):
    def __init__(self, alpha = 0.75 , gamma = 2.0, num_neg = 10000, num_hard = 100, ratio = 100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_neg = num_neg
        self.num_hard = num_hard
        self.ratio = ratio
    
    def forward(self, pred, target):
        classification_losses = []
        batch_size = pred.shape[0]
        
        for j in range(batch_size):
            pred_b = pred[j]
            target_b = target[j]
    
            cls_prob = torch.sigmoid(pred_b.detach())
            cls_prob = torch.clamp(cls_prob, 1e-4, 1.0 - 1e-4)
            
            alpha_factor = torch.ones(pred_b.shape).to(pred_b.device) * self.alpha
            alpha_factor = torch.where(torch.eq(target_b, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(target_b, 1.), 1. - cls_prob, cls_prob)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

            bce = F.binary_cross_entropy_with_logits(pred_b, target_b, reduction='none')
            num_positive_pixels = torch.sum(target_b == 1)
            cls_loss = focal_weight * bce
           
            record_targets = target_b.clone()
            
            if num_positive_pixels > 0:
                FN_weights = 4.0  # 10.0  for ablation study
                FN_index = torch.lt(cls_prob, 0.8) & (record_targets == 1)  # 0.9
                cls_loss[FN_index == 1] = FN_weights * cls_loss[FN_index == 1]
                Negative_loss = cls_loss[record_targets == 0]
                Positive_loss = cls_loss[record_targets == 1]
                neg_idcs = random.sample(range(len(Negative_loss)), min(self.num_neg, len(Negative_loss))) 
                Negative_loss = Negative_loss[neg_idcs] 
                _, keep_idx = torch.topk(Negative_loss, min(self.ratio * num_positive_pixels, Negative_loss.shape[0])) 
                Negative_loss = Negative_loss[keep_idx] 
                Positive_loss = Positive_loss.sum()
                Negative_loss = Negative_loss.sum()
                cls_loss = Positive_loss + Negative_loss

            else:
                Negative_loss = cls_loss[record_targets == 0]
                neg_idcs = random.sample(range(len(Negative_loss)), min(self.num_neg, len(Negative_loss)))
                Negative_loss = Negative_loss[neg_idcs]
                assert len(Negative_loss) > self.num_hard
                _, keep_idx = torch.topk(Negative_loss, self.num_hard)
                Negative_loss = Negative_loss[keep_idx]
                Negative_loss = Negative_loss.sum()
                cls_loss = Negative_loss
                
            classification_losses.append(cls_loss / torch.clamp(num_positive_pixels.float(), min=1.0))
            
        return torch.mean(torch.stack(classification_losses))

class DetectionLoss(nn.Module):
    def __init__(self, model, device, crop_size, tal_topk=7, tal_counter=30, cls_weight=0.5, box_weight=7.5, dfl_weight=1.5) -> None:
        super().__init__()
     
        self.reg_max = model.head.reg_max
        self.nc = model.head.cls
        self.use_dfl = self.reg_max  > 1
        
        self.device = device
        self.assigner = TaskAlignedAssigner(topk=tal_topk, tal_counter=tal_counter, num_classes=self.nc, alpha=2.0, beta=0.5) # we want to care cls score more 
        self.proj = torch.arange(self.reg_max, dtype=torch.float).to(device)

        self.crop_size = crop_size
        # self.cls_loss = nn.BCEWithLogitsLoss(reduction='none') # or focal loss 
        self.focal_loss = VarientFocalLoss()
        
        self.bbox_loss = BboxLoss(self.reg_max - 1, use_dfl=self.use_dfl).to(device)
        
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.dfl_weight = dfl_weight
        
    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 6, c // 6).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)
    
    def increase_counter(self):
        self.assigner.increase_counter()
    
    def forward(self, preds, targets):
        loss = torch.zeros(3, device=self.device) 
        feat = preds[1]
                
        pred_scores = preds[0].view(feat.shape[0], 1, -1).permute(0, 2, 1).contiguous()
        pred_distri = preds[1].view(feat.shape[0], self.reg_max * 6, -1).permute(0, 2, 1).contiguous()
        
        bs = pred_scores.shape[0]      

        anchor_points, stride_tensor = make_anchors(feat, self.crop_size, 0.5)
        
        gt_labels, gt_bboxes = targets.split((1, 6), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri) 
        
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        
        target_scores_sum = max(target_scores.sum(), 1)
        
        loss[1] = self.focal_loss(pred_scores, target_scores)
        
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)
        
        loss[0] *= self.box_weight
        loss[1] *= self.cls_weight
        loss[2] *= self.dfl_weight
        
        return loss.sum() * bs, loss.detach()