import torch
import torch.nn as nn
import torchvision.models as models

class ConceptExtractor(nn.Module):
    """
    Extracts continuous concept predictions from images using a pre-trained backbone.
    """
    def __init__(self, num_concepts: int, backbone_name: str = 'resnet18', pretrained: bool = True):
        super().__init__()
        
        if backbone_name == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_concepts)
        else:
            raise NotImplementedError(f"Backbone {backbone_name} is not implemented.")
            
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        return self.sigmoid(logits)

class LabelPredictor(nn.Module):
    """
    Predicts final class logits strictly from the predicted concepts.
    """
    def __init__(self, num_concepts: int, num_classes: int, hidden_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_concepts, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, c_hat: torch.Tensor) -> torch.Tensor:
        return self.mlp(c_hat)

class ConceptBottleneckModel(nn.Module):
    """
    Complete CBM architecture combining the ConceptExtractor and LabelPredictor.
    """
    def __init__(self, num_concepts: int, num_classes: int, backbone_name: str = 'resnet18', pretrained: bool = True):
        super().__init__()
        self.extractor = ConceptExtractor(num_concepts=num_concepts, backbone_name=backbone_name, pretrained=pretrained)
        self.predictor = LabelPredictor(num_concepts=num_concepts, num_classes=num_classes)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns both class logits (y_hat) and concept predictions (c_hat).
        """
        c_hat = self.extractor(x)
        y_hat = self.predictor(c_hat)
        return y_hat, c_hat
