from abc import ABC, abstractmethod
import os
import numpy as np
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
    
@register_feature_extractor("one-hot")
class OneHotFeatureExtractor(FeatureExtractor):
    def __init__(self, device, data_dir):
        super().__init__()
        self.path = os.path.join(data_dir, "identity_features")
        self.dim = 4 # Temp adjustment
        self.device = device

    def extract_features(self, frame, metadata):
        file_name = metadata["file_path"].split("/")[-1]
        stem = os.path.splitext(file_name)[0]
        feats = np.load(os.path.join(self.path, f"{stem}.npy"))
        feats = torch.from_numpy(feats).float().to(self.device)
        print(feats.shape)
        return feats
    
@register_feature_extractor("feature-map")
class FeatureMapFeatureExtractor(FeatureExtractor):
    def __init__(self, device, data_dir, feature_dir):
        super().__init__()
        # self.path = os.path.join(data_dir, "identity_features")
        self.device = device
        self.feature_dir = feature_dir
        npys = [f for f in os.listdir(feature_dir) if f.endswith('.npy')]
        # Load one to find the dimension
        if npys:
            sample_feats = np.load(os.path.join(feature_dir, npys[0]))
            self.dim = sample_feats.shape[-1]
        else:
            raise ValueError(f"No .npy files found in {feature_dir}")

    def extract_features(self, frame, metadata):
        file_name = metadata["file_path"].split("/")[-1]
        stem = os.path.splitext(file_name)[0]
        feats = np.load(os.path.join(self.feature_dir, f"{stem}.npy"))
        feats = torch.from_numpy(feats).float().to(self.device)
        # feats = feats / torch.norm(feats, dim=1, keepdim=True)
        # print(feats.shape)
        return feats
    

@register_feature_extractor("lang-splat")
class LangSplatFeatureExtractor(FeatureExtractor):
    def __init__(self, device, data_dir, feature_dir, level=0):
        super().__init__()
        # self.path = os.path.join(data_dir, "identity_features")
        self.device = device
        self.feature_dir = feature_dir
        self.level = level
        npys = [f for f in os.listdir(feature_dir) if f.endswith('.npy')]
        # Load one to find the dimension
        if npys:
            sample_feats = np.load(os.path.join(feature_dir, npys[0]))
            self.dim = sample_feats.shape[-1]
        else:
            raise ValueError(f"No .npy files found in {feature_dir}")
        
    def set_level(self, level):
        self.level = level

    def _create_feature_map(self, features, segments, level):
        seg_map = segments[level] # (H, W)
        mask = seg_map != -1
        feature_map = np.zeros((seg_map.shape[0], seg_map.shape[1], features.shape[1]), dtype=np.float32)
        feature_map[mask] = features[seg_map[mask]]
        return feature_map

    def extract_features(self, frame, metadata):
        file_name = metadata["file_path"].split("/")[-1]
        stem = os.path.splitext(file_name)[0]
        features_path = os.path.join(self.feature_dir, f"{stem}_f.npy")
        segments_path = os.path.join(self.feature_dir, f"{stem}_s.npy")
        features = np.load(features_path) # (N, D)
        segments = np.load(segments_path) # (4, H, W)
        return self._create_feature_map(features, segments, self.level)