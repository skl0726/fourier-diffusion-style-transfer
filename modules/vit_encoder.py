"""
Vision-Transformer Style Encoder
--------------------------------
Requirements:
  pip install timm torch torchvision pillow

Outputs:
  Tensor shape (1, N_style, 768)  [dtype=float32, device=given]
"""

# 이 코드는 encoder를 몇 번 거치는지?? encoder를 몇 번 거쳐야 효과가 있는지?

import torch, torch.nn as nn
from PIL import Image
from torchvision import transforms as T
import timm


class ViTStyleEncoder(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", pretrained=True):
        super().__init__()
        self.vit = timm.create_model(model_name,
                                     pretrained=pretrained,
                                     num_classes=0,  # remove head
                                     global_pool="")  # keep tokens
        self.norm = nn.LayerNorm(self.vit.num_features)

    @torch.no_grad()
    def forward(self, pil_img: Image.Image, device="cpu"):
        self._to(device).eval()

        tfm = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        x = tfm(pil_img.convert("RGB")).unsqueeze(0).to(device)  # (1,3,224,224)

        tokens = self.vit.patch_embed(x)          # (1, N+1, 768) incl. class token
        cls_tok, patch_tok = tokens[:, :1], tokens[:, 1:]
        patch_tok = self.norm(patch_tok)          # (1, N_style, 768)

        # zero-mean/var-1 정규화 → 스타일 강도 균일화
        patch_tok = patch_tok / patch_tok.std(dim=-1, keepdim=True)

        MAX_TOKEN = 77
        patch_tok = patch_tok[:, :MAX_TOKEN]
        return patch_tok                          # (1, N_style, 768)

    def _to(self, device):
        self.vit = self.vit.to(device)
        self.norm = self.norm.to(device)
        return self