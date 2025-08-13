from abc import ABC, abstractmethod
import torch
from lseg import LSegNet

# registry.py
BACKPROJECTION_FEATURE_EXTRACTORS = {}

def register_feature_extractor(name):
    def decorator(cls):
        BACKPROJECTION_FEATURE_EXTRACTORS[name] = cls
        return cls
    return decorator

def get_feature_extractor(name, device, **kwargs):
    if name not in BACKPROJECTION_FEATURE_EXTRACTORS:
        raise ValueError(f"Unknown feature extractor: {name}. "
                         f"Available: {list(BACKPROJECTION_FEATURE_EXTRACTORS.keys())}")
    return BACKPROJECTION_FEATURE_EXTRACTORS[name](device=device, **kwargs)

class FeatureExtractor(ABC):
    def extract_features(self, frame):
        raise NotImplementedError("This method should be overridden by subclasses")
    
@register_feature_extractor("lseg")
class LSegFeatureExtractor(FeatureExtractor):
    def __init__(self, device):
        super().__init__()
        # Initialize the LSeg model here

        net = LSegNet(
            backbone="clip_vitl16_384",
            features=256,
            crop_size=480,
            arch_option=0,
            block_depth=0,
            activation="lrelu",
        )
        # Load pre-trained weights
        net.load_state_dict(
            torch.load("./checkpoints/lseg_minimal_e200.ckpt", map_location=device)
        )
        net.eval()
        net.to(device)

        self.device = device
        self.net = net
        self.dim = 512

    def extract_features(self, frame):
        with torch.no_grad():
            # Implement feature extraction logic for LSeg
            height, width = frame.shape[:2]
            frame = torch.nn.functional.interpolate(
                        frame[None].permute(0, 3, 1, 2).to(self.device),
                        size=(480, 480),
                        mode="bilinear",
                    )
            frame.to(self.device)
            feats = self.net.forward(frame)
            feats = torch.nn.functional.normalize(feats, dim=1)
        feats = torch.nn.functional.interpolate(
            feats, size=(height, width), mode="bilinear"
        )[0]
        feats = feats.permute(1, 2, 0)
        return feats

@register_feature_extractor("dino")
class DinoFeatureExtractor(FeatureExtractor):
    def __init__(self, device):
        super().__init__()
        # Initialize the LSeg model here

        feature_extractor = (
            torch.hub.load("facebookresearch/dinov2:main", "dinov2_vitl14_reg")
            .to(device)
            .eval()
        )

        dinov2_vits14_reg = feature_extractor

        self.device = device
        self.net = dinov2_vits14_reg
        self.dim = 1024

    def extract_features(self, frame):
        with torch.inference_mode():
            # Implement feature extraction logic for LSeg
            height, width = frame.shape[:2]
            output = torch.nn.functional.interpolate(
                frame[None].permute(0, 3, 1, 2).cuda(),
                size=(224 * 4, 224 * 4),
                mode="bilinear",
                align_corners=False,
            )
            feats = self.net.forward_features(output)["x_norm_patchtokens"]
            feats = feats[0].reshape((16 * 4, 16 * 4, self.dim))
        feats = torch.nn.functional.interpolate(
            feats.unsqueeze(0).permute(0, 3, 1, 2),
            size=(height, width),
            mode="nearest",
        )[0]
        feats = feats.permute(1, 2, 0)
        return feats