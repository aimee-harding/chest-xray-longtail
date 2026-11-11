import torch
import torch.nn as nn
from torchvision import models

#creates pytorch model
class DenseNet121Backbone(nn.Module):
    #constructor 
    def __init__(self, num_classes: int = 14, pretrained: bool = True):
        super().__init__()

        # load pretrained DenseNet-121
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        densenet = models.densenet121(weights=weights)

        # extract the feature extractor (everything except the classifier)
        self.features = densenet.features

        # get the feature dimension
        num_features = densenet.classifier.in_features  # 1024

        # replace classifier head, with 1024 -> 14
        self.classifier = nn.Linear(num_features, num_classes)

        # initialise new head
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        print(f"[DenseNet121Backbone] Initialised with {num_features} features "
              f"→ {num_classes} classes "
              f"({'pretrained' if pretrained else 'random'} weights)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input x: tensor of shape (B, 3, 224, 224)
        # returns logits: tensor ofshape (B, 14)
        
        # feature extraction 
        # pass through all DenseNet conv/dense blocks
        features = self.features(x)          # (B, 1024, 7, 7)

        # global average pooling 
        out = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1)) # (B, 1024, 1, 1)
        out = torch.flatten(out, 1)  # (B, 1024)

        # classification head 
        logits = self.classifier(out)        # (B, 14)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        #returns 1024-dim feature vector without passing through classifier head.
        #for linear probing evals
        #input x: tensor of shape (B, 3, 224, 224)
        #returns: tensor of shape (B, 1024)

        features = self.features(x)
        out = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        return out


# utility functions 

def get_backbone(num_classes: int = 14, pretrained: bool = True) -> DenseNet121Backbone:
    return DenseNet121Backbone(num_classes=num_classes, pretrained=pretrained)


def freeze_backbone(model: DenseNet121Backbone) -> None:
    #freeze all parameters except the classifier head
    #used for classifier retraining (stage 2)

    # freeze the feature extractor
    for param in model.features.parameters():
        param.requires_grad = False  #pytorch doesnt calc gradients for these params

    # keep classifier head trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    n_frozen    = sum(p.numel() for p in model.features.parameters())
    n_trainable = sum(p.numel() for p in model.classifier.parameters())
    print(f"[freeze_backbone] Frozen: {n_frozen:,} params | "
          f"Trainable: {n_trainable:,} params")


def unfreeze_backbone(model: DenseNet121Backbone) -> None:
    #unfreeze all parameters for end-to-end training

    for param in model.parameters():
        param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters())
    print(f"[unfreeze_backbone] All {n_trainable:,} params trainable")


def count_parameters(model: nn.Module) -> dict:
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total     = trainable + frozen
    return {
        "trainable": trainable,
        "frozen":    frozen,
        "total":     total,
    }