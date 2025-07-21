import torch.nn as nn
import torch.nn.functional as F
import torch

class MultiTaskHeadMixin:
    def init_attr_heads(self, in_channels, num_clf_heads=8, clf_classes=5):
        self.ph_head = nn.Linear(in_channels, 1)
        self.vbn_head = nn.Linear(in_channels, 1)
        self.clf_heads = nn.ModuleList([
            nn.Linear(in_channels, clf_classes) for _ in range(num_clf_heads)
        ])

    def forward_attr(self, feat):
        pooled_feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0), -1)
        ph = torch.sigmoid(self.ph_head(pooled_feat)) * 14.0
        vbn = torch.sigmoid(self.vbn_head(pooled_feat)) * 50.0
        clf = torch.stack([h(pooled_feat) for h in self.clf_heads], dim=1)  # [B,8,5]
        return ph, vbn, clf
