import torch
from src.models.prof_fpn_segmenter import ResNet50_FPN_Segmenter

def main():
    model = ResNet50_FPN_Segmenter(num_classes=11, pretrained=True)
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    print("logits:", y.shape, y.dtype)  # expected: [2, 11, 512, 512]

if __name__ == "__main__":
    main()