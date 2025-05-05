# models/architectures/vit.py
from torchvision.models import vit_b_16

def get_vit(pretrained=True, num_classes=10):
    model = vit_b_16(pretrained=pretrained)
    model.heads.head.out_features = num_classes
    return model