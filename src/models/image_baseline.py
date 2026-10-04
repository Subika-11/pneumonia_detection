import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

class DenseNet121Baseline(nn.Module):
    """
    Pretrained DenseNet-121 transfer learning baseline for chest X-ray classification.
    
    DenseNet-121 is chosen as the primary radiological baseline because:
    1. Dense connections between layers encourage extensive feature reuse across scales.
    2. Alleviates vanishing gradients when learning subtle pulmonary opacities.
    3. Has demonstrated state-of-the-art performance in landmark medical imaging literature
       (CheXNet, Stanford CheXpert, MIMIC-CXR baselines).
    """
    def __init__(self, pretrained=True, freeze_backbone=True, dropout=0.2):
        super(DenseNet121Baseline, self).__init__()
        
        # Load weights
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.densenet = densenet121(weights=weights)
        
        # In DenseNet121, features.norm5 output has 1024 channels
        in_features = self.densenet.classifier.in_features  # 1024
        
        # Replace the 1000-class ImageNet classifier with our clinical binary head
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1)
        )
        
        if freeze_backbone:
            self.freeze_backbone()
            
    def forward(self, x):
        """
        Forward pass.
        Args:
            x (Tensor): [Batch, 3, 224, 224] normalized image tensor
        Returns:
            logits (Tensor): [Batch, 1] raw logit scores (before sigmoid)
        """
        return self.densenet(x)
        
    def extract_features(self, x):
        """
        Extracts 1024-dimensional feature embedding before classification.
        Useful for multimodal fusion (Phase 7).
        """
        features = self.densenet.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out
        
    def freeze_backbone(self):
        """Freezes all layers except the new classifier head."""
        for param in self.densenet.features.parameters():
            param.requires_grad = False
        for param in self.densenet.classifier.parameters():
            param.requires_grad = True
            
    def unfreeze_backbone(self):
        """Unfreezes all layers in the backbone for fine-tuning."""
        for param in self.densenet.parameters():
            param.requires_grad = True
            
    def unfreeze_denseblock4(self):
        """Unfreezes only DenseBlock4 and the transition layer for targeted fine-tuning."""
        for name, param in self.densenet.features.named_parameters():
            if 'denseblock4' in name or 'norm5' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for param in self.densenet.classifier.parameters():
            param.requires_grad = True
            
    def get_parameter_summary(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params
        }
