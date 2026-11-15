"""Frozen image encoders: image tensors -> one feature vector per observation."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import nn

DIMENSIONS = {"resnet18": 512, "dinov2": 384, "convnext_tiny": 768}
DINO_MODEL = "facebook/dinov2-small"
DINO_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
FILL = (124, 116, 104)


def preprocess(image: Image.Image) -> torch.Tensor:
    """Shared full-frame letterbox, not the weights' default center crop."""
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        image = Image.alpha_composite(Image.new("RGBA", image.size, (*FILL, 255)),
                                      image.convert("RGBA")).convert("RGB")
    else:
        image = image.convert("RGB")
    scale = 224 / max(image.size)
    width, height = [max(1, round(v * scale)) for v in image.size]
    canvas = Image.new("RGB", (224, 224), FILL)
    canvas.paste(image.resize((width, height), Image.Resampling.BILINEAR),
                 ((224 - width) // 2, (224 - height) // 2))
    pixels = np.asarray(canvas, dtype=np.float32) / 255.0
    pixels = (pixels - np.asarray(MEAN, dtype=np.float32)) / np.asarray(STD, dtype=np.float32)
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy())


class FrozenEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, name: str, source: str):
        super().__init__()
        self.backbone = backbone
        self.name = name
        self.dimension = DIMENSIONS[name]
        self.source = source
        self.requires_grad_(False)
        self.eval()

    @torch.inference_mode()
    def forward(self, images):
        self.eval()  # Keep BatchNorm statistics fixed, even after an accidental train().
        if self.name == "dinov2":
            # Final layer-normalized CLS token, not pooler/classifier probabilities.
            features = self.backbone(pixel_values=images).last_hidden_state[:, 0, :]
        else:
            features = self.backbone(images)
        if features.ndim != 2 or features.shape[1] != self.dimension:
            raise RuntimeError("Unexpected encoder feature dimensions.")
        return features


def build_encoder(name: str, *, pretrained: bool = True) -> FrozenEncoder:
    """pretrained=False is for synthetic offline tests only, never the extraction CLI."""
    if name == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None, progress=False)
        model.fc = nn.Identity()
        source = "torchvision:ResNet18_Weights.IMAGENET1K_V1"
    elif name == "convnext_tiny":
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None, progress=False)
        model.classifier[-1] = nn.Identity()  # Retain the pretrained LayerNorm and flatten.
        source = "torchvision:ConvNeXt_Tiny_Weights.IMAGENET1K_V1"
    elif name == "dinov2":
        from transformers import Dinov2Config, Dinov2Model
        if pretrained:
            model = Dinov2Model.from_pretrained(DINO_MODEL, revision=DINO_REVISION,
                                                use_safetensors=True, attn_implementation="eager")
        else:
            model = Dinov2Model(Dinov2Config(hidden_size=384, num_hidden_layers=12,
                num_attention_heads=6, intermediate_size=1536, image_size=224, patch_size=14))
        source = f"huggingface:{DINO_MODEL}@{DINO_REVISION}"
    else:
        raise ValueError("Unknown encoder name.")
    return FrozenEncoder(model, name, source if pretrained else "SYNTHETIC_TEST_RANDOM_WEIGHTS")
