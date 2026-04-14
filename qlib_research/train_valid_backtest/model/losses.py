import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseRankingLoss(nn.Module):
    def __init__(self, margin=1.0, use_sigmoid=True):
        super().__init__()
        self.margin = margin
        self.use_sigmoid = use_sigmoid

    def forward(self, scores, labels):
        scores = scores.view(-1)
        labels = labels.view(-1)
        batch_size = scores.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=scores.device)

        scores_i = scores.unsqueeze(1)
        scores_j = scores.unsqueeze(0)
        labels_i = labels.unsqueeze(1)
        labels_j = labels.unsqueeze(0)

        label_diff = labels_i - labels_j
        mask = label_diff > 0

        if mask.sum() == 0:
            return torch.tensor(0.0, device=scores.device)

        score_diff = scores_i - scores_j

        if self.use_sigmoid:
            pairwise_loss = F.softplus(-score_diff)
        else:
            pairwise_loss = F.relu(self.margin - score_diff)

        loss = (pairwise_loss * mask.float()).sum() / mask.sum()
        return loss


def listwise_ranking_loss(scores, labels, temperature=1.0):
    scores = scores.view(-1)
    labels = labels.view(-1)
    label_probs = F.softmax(labels / temperature, dim=0)
    # Use log_softmax for better numerical stability than F.softmax().log()
    log_score_probs = F.log_softmax(scores / temperature, dim=0)
    loss = F.kl_div(log_score_probs, label_probs, reduction="batchmean")
    return loss


def pearson_corr_loss(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    pred = pred - pred.mean()
    target = target - target.mean()
    cov = (pred * target).mean()
    std_pred = pred.std(correction=0) + 1e-8
    std_target = target.std(correction=0) + 1e-8
    return 1 - cov / (std_pred * std_target)


class HuberPearsonLoss(nn.Module):
    def __init__(self, huber_delta=1.0, pearson_weight=0.1):
        super().__init__()
        self.huber = nn.HuberLoss(delta=huber_delta)
        self.pearson_weight = pearson_weight

    def forward(self, pred, target):
        huber_loss = self.huber(pred, target)
        pearson_loss = pearson_corr_loss(pred, target)
        return huber_loss + self.pearson_weight * pearson_loss
