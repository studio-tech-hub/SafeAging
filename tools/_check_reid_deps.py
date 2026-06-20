import torchvision, torch
print("torchvision:", torchvision.__version__)
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch.nn as nn
m = mobilenet_v3_small(weights=None)
feat = nn.Sequential(m.features, m.avgpool, nn.Flatten(1))
feat.eval()
t = torch.zeros(1, 3, 224, 224)
with torch.no_grad():
    out = feat(t)
print("embedding_dim:", out.shape[1])
print("MobileNetV3-Small OK")
