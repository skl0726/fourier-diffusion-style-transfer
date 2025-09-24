import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import CLIPVisionModel
from PIL import Image


class StyleEncoder(nn.Module):
    def __init__(self,
                 device,
                 heads=8,
                 dim_head=64,
                 sty_alpha: float = 0.1,
                 clip_model_name: str = "openai/clip-vit-base-patch32",
                 freeze_clip: bool = True):
        super().__init__()
        self.sty_alpha = sty_alpha
        self.device = device

        self.clip = CLIPVisionModel.from_pretrained(clip_model_name).to(self.device)
        self.clip.eval()
        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False

        clip_feat_dim = self.clip.config.hidden_size
        inner_dim = dim_head * heads
        self.norm = nn.LayerNorm(clip_feat_dim) # optional
        self.to_k_injected = nn.Linear(clip_feat_dim, inner_dim, bias=False)
        self.to_v_injected = nn.Linear(clip_feat_dim, inner_dim, bias=False)

        self.preprocess = transforms.Compose([
            transforms.Resize(self.clip.config.image_size if hasattr(self.clip.config, "image_size") else 224),
            transforms.CenterCrop(self.clip.config.image_size if hasattr(self.clip.config, "image_size") else 224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711))
        ])

    def normalize_batch_tensor(self, images: torch.Tensor):
        if images.dtype == torch.uint8:
            images = images.float() / 255.0 # if input in 0..255 convert to 0..1
        images = images.to(self.device)

        target_size = self.clip.config.image_size if hasattr(self.clip.config, "image_size") else 224
        B, C, H, W = images.shape
        if H != target_size or W != target_size:
            # Note: torchvision transforms operate on PIL images or batched tensors differently; here we do simple resizing via F.interpolate if needed
            images = F.interpolate(images, size=(target_size, target_size), mode='bilinear', align_corners=False)

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).view(1, C, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).view(1, C, 1, 1)
        images = (images - mean) / std

        return images
    
    @torch.no_grad()
    def extract_clip_patch_features(self, images):
        if isinstance(images, Image.Image):
            images = [images]

        if isinstance(images, (list, tuple)) and isinstance(images[0], Image.Image):
            proc = []
            for img in images:
                t = self.preprocess(img) # ToTensor + Normalize
                proc.append(t)
            batch = torch.stack(proc, dim=0).to(self.device)
        elif isinstance(images, torch.Tensor):
            batch = self.normalize_batch_tensor(images)

        outputs = self.clip(pixel_values=batch)
        feats = outputs.last_hidden_size
        if feats.shape[1] > 1:
            feats = feats[:, 1:, :] # remove cls token if present
        
        return feats

    def forward(self, x):
        feats = self.extract_clip_patch_features(x)
        B, seq, feat_dim = feats.shape

        feats = self.norm(feats)

        k_proj = self.to_k_injected(feats)
        v_proj = self.to_v_injected(feats)

        k_proj = k_proj.view(B, seq, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous() # [B, seq_s, heads, dim_head] -> [B, heads, seq_s, head_dim]
        v_proj = v_proj.view(B, seq, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous() # [B, seq_s, heads, dim_head] -> [B, heads, seq_s, head_dim]

        return {'k': k_proj, 'v': v_proj, 'sty_alpha': self.sty_alpha}


# import torch, torch.nn as nn
# from PIL import Image
# from transformers import CLIPVisionModel, CLIPImageProcessor


# _clip_model = None
# _clip_proc = None
# _proj = None


# def clip_vit_encoder(img, # PIL image or torch.Tensor
#                      model_name="openai/clip-vit-large-patch14",
#                      out_dim=768,
#                      max_tokens=64,
#                      device="cuda"):
#     global _clip_model, _clip_proc, _proj
#     device = torch.device(device)

#     if _clip_model is None:
#         _clip_model = CLIPVisionModel.from_pretrained(model_name).to(device).eval()
#         _clip_proc = CLIPImageProcessor.from_pretrained(model_name)
#         hidden = _clip_model.config.hidden_size  # e.g., 1024 for ViT-L/14
#         _proj = nn.Identity() if hidden == out_dim else nn.Linear(hidden, out_dim, bias=False).to(device)
#         for p in list(_clip_model.parameters()) + list(_proj.parameters()):
#             p.requires_grad_(False)

#     if isinstance(img, torch.Tensor):
#         if img.min() < 0:
#             img = (img + 1.0) / 2.0
#         img = img.clamp(0.0, 1.0)
        
#         if img.dim() == 4:
#             img_list = []
#             for i in range(img.shape[0]):
#                 img_np = img[i].permute(1, 2, 0).cpu().numpy()
#                 img_np = (img_np * 255).astype('uint8')
#                 pil_img = Image.fromarray(img_np)
#                 img_list.append(pil_img)
#             img = img_list
#         elif img.dim() == 3:
#             img_np = img.permute(1, 2, 0).cpu().numpy()
#             img_np = (img_np * 255).astype('uint8')
#             img = Image.fromarray(img_np)

#     inputs = _clip_proc(images=img, return_tensors="pt")
#     clip_input = inputs["pixel_values"].to(device)

#     with torch.no_grad():
#         out = _clip_model(pixel_values=clip_input)
#         tokens = out.last_hidden_state[:, 1:, :]  # drop CLS
#         B, N, C = tokens.shape
#         if max_tokens is not None and N > max_tokens:
#             factor = N // max_tokens
#             tokens = tokens[:, :factor * max_tokens, :]
#             tokens = tokens.reshape(B, max_tokens, factor, C).mean(dim=2)
#         tokens = _proj(tokens)  # (1, N_style, out_dim)

#     return tokens


# class StyleSelfAttention(nn.Module):
#     def __init__(self, dim=768, nhead=8, nlayers=2, dim_feedforward=2048, dropout=0.1):
#         super().__init__()
#         encoder_layer = nn.TransformerEncoderLayer(d_model=dim,
#                                                    nhead=nhead,
#                                                    dim_feedforward=dim_feedforward,
#                                                    dropout=dropout,
#                                                    batch_first=True)
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

#     def forward(self, tokens):
#         return self.encoder(tokens)


if __name__ == "__main__":
    model = StyleEncoder(device=torch.device("cpu"))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total params:", total_params)
    print("Trainable params:", trainable_params)


"""
Total params: 11027968
Trainable params: 11027968
"""