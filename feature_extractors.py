from abc import ABC, abstractmethod
import os
import cv2
import numpy as np
import torch
from lseg import LSegNet
import clip

# registry
BACKPROJECTION_FEATURE_EXTRACTORS = {}


def register_feature_extractor(name):
    def decorator(cls):
        BACKPROJECTION_FEATURE_EXTRACTORS[name] = cls
        return cls

    return decorator


def get_feature_extractor(name, device, **kwargs):
    if name not in BACKPROJECTION_FEATURE_EXTRACTORS:
        raise ValueError(
            f"Unknown feature extractor: {name}. "
            f"Available: {list(BACKPROJECTION_FEATURE_EXTRACTORS.keys())}"
        )
    return BACKPROJECTION_FEATURE_EXTRACTORS[name](device=device, **kwargs)


class FeatureExtractor(ABC):
    def extract_features(self, frame, metadata):
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

    def extract_features(self, frame, metadata=None):
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

    def extract_features(self, frame, metadata=None):
        with torch.inference_mode():
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
        self.dim = 4  # Temp adjustment
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
        self.device = device
        self.feature_dir = feature_dir
        npys = [f for f in os.listdir(feature_dir) if f.endswith(".npy")]
        # Load one to find the dimension
        if npys:
            sample_feats = np.load(os.path.join(feature_dir, npys[0]))
            self.dim = sample_feats.shape[-1]
        else:
            raise ValueError(f"No .npy files found in {feature_dir}")

    def extract_features(self, frame, metadata):
        if "file_path" in metadata:
            file_name = metadata["file_path"].split("/")[-1]
        else:
            file_name = metadata["image_name"].split("/")[-1]
        stem = os.path.splitext(file_name)[0]
        feats = np.load(os.path.join(self.feature_dir, f"{stem}.npy"))
        feats = torch.from_numpy(feats).float().to(self.device)
        return feats


@register_feature_extractor("lang-splat")
class LangSplatFeatureExtractor(FeatureExtractor):
    def __init__(self, device, data_dir, feature_dir, level=0):
        super().__init__()
        self.device = device
        self.feature_dir = feature_dir
        self.level = level
        npys = [f for f in os.listdir(feature_dir) if f.endswith("_f.npy")]
        # Load one to find the dimension
        if npys:
            sample_feats = np.load(os.path.join(feature_dir, npys[0]))
            self.dim = sample_feats.shape[-1]
        else:
            raise ValueError(f"No .npy files found in {feature_dir}")

    def set_level(self, level):
        self.level = level

    def _create_feature_map(self, features, segments, level):
        seg_map = segments[level]  # (H, W)
        mask = seg_map != -1
        feature_map = np.zeros(
            (seg_map.shape[0], seg_map.shape[1], features.shape[1]), dtype=np.float32
        )
        feature_map[mask] = features[seg_map[mask]]
        return feature_map

    def extract_features(self, frame, metadata):
        file_name = metadata["image_name"].split("/")[-1]
        stem = os.path.splitext(file_name)[0]
        features_path = os.path.join(self.feature_dir, f"{stem}_f.npy")
        segments_path = os.path.join(self.feature_dir, f"{stem}_s.npy")
        features = np.load(features_path)  # (N, D)
        segments = np.load(segments_path).astype(int)  # (4, H, W)
        feature_map = self._create_feature_map(features, segments, self.level)
        return torch.from_numpy(feature_map).float().to(self.device)

@register_feature_extractor("scannet-instance")
class ScanNetInstanceFeatureExtractor(FeatureExtractor):
    def __init__(self, device, data_dir, feature_dir, level=0,max_classes=100):
        super().__init__()
        self.device = device
        self.feature_dir = feature_dir
        self.level = level
        # Find the name of data_dir
        scene_name = os.path.basename(os.path.normpath(data_dir))
        self.scene_name = scene_name
        self.max_classes = max_classes
        self.dim = max_classes  # One-hot encoding dimension

    def set_level(self, level):
        self.level = level

    def _create_feature_map(self, features, segments, level):
        seg_map = segments[level]  # (H, W)
        mask = seg_map != -1
        feature_map = np.zeros(
            (seg_map.shape[0], seg_map.shape[1], features.shape[1]), dtype=np.float32
        )
        feature_map[mask] = features[seg_map[mask]]
        return feature_map

    def extract_features(self, frame, metadata):
        file_name = metadata["image_name"].split("/")[-1]
        stem = os.path.splitext(file_name)[0]
        features_path = os.path.join(self.feature_dir, f"{stem}.png")
        feature_map = cv2.imread(features_path, cv2.IMREAD_UNCHANGED)
        feature_map_one_hot = np.zeros((feature_map.shape[0], feature_map.shape[1], self.max_classes), dtype=np.float32)
        eye = np.eye(self.max_classes)
        feature_map_one_hot = eye[feature_map]

        return torch.from_numpy(feature_map_one_hot).float().to(self.device)
    

@register_feature_extractor("one-hot-3dovs")
class Ovs3DOneHotFeatureExtractor(FeatureExtractor):
    def __init__(self, device, data_dir):
        super().__init__()
        self.device = device
        self.feature_dir = os.path.join(data_dir, "segmentations")
        classes_path = os.path.join(self.feature_dir, "classes.txt")
        with open(classes_path, 'r') as f:
            classes = f.readlines()
        self.classes = sorted([c.strip() for c in classes])

        self.dim = len(self.classes)
        
        # seg_directories
        seg_directories = [d for d in os.listdir(self.feature_dir) if os.path.isdir(os.path.join(self.feature_dir, d))]
        seg_directories = sorted(seg_directories)
        self.train_dirs = seg_directories[:-1] # Not exactly train dirs though
        self.test_dir = seg_directories[-1]


    def _create_feature_map(self, features_path):

        eye = torch.eye(self.dim)
        feature_map = None
        for index, class_ in enumerate(self.classes):
            mask_path = os.path.join(features_path, f"{class_}.png")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
            mask_height, mask_width = mask.shape
            mask = cv2.resize(mask, (mask_width // 4, mask_height // 4)) # TODO: Change hardcoding
            if feature_map is None:
                feature_map = torch.zeros((mask.shape[0], mask.shape[1], self.dim), dtype=torch.float32)
            feature_map[mask > 128] = eye[index]  # Assuming mask is binary with values 0 and 255
        return feature_map

    def extract_features(self, frame, metadata):
        file_name = metadata["image_name"].split("/")[-1]
        stem = os.path.splitext(file_name)[0]
        features_path = os.path.join(self.feature_dir, f"{stem}")
        if not os.path.exists(features_path):
            return None
        
        feats = self._create_feature_map(features_path)
        return feats
        

@register_feature_extractor("clip-text-3dovs")
class OvsClipTextFeatureExtractor(FeatureExtractor):
    def __init__(self, device, data_dir):
        super().__init__()
        self.device = device
        self.feature_dir = os.path.join(data_dir, "segmentations")
        classes_path = os.path.join(self.feature_dir, "classes.txt")
        with open(classes_path, 'r') as f:
            classes = f.readlines()
        self.classes = sorted([c.strip() for c in classes])

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
        
        clip_text_encoder = net.clip_pretrained.encode_text

        del net

        prompts = self.classes

        embeddings = clip_text_encoder(clip.tokenize(prompts).to(device)).float()

        self.text_embeddings = embeddings

        # seg_directories
        seg_directories = [d for d in os.listdir(self.feature_dir) if os.path.isdir(os.path.join(self.feature_dir, d))]
        seg_directories = sorted(seg_directories)
        self.train_dirs = seg_directories[:-1] # Not exactly train dirs though
        self.test_dir = seg_directories[-1]


    def _create_feature_map(self, features_path):

        # eye = torch.eye(self.dim)
        feature_map = None
        for index, class_ in enumerate(self.classes):
            mask_path = os.path.join(features_path, f"{class_}.png")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
            mask_height, mask_width = mask.shape
            mask = cv2.resize(mask, (mask_width // 4, mask_height // 4)) # TODO: Change hardcoding
            if feature_map is None:
                feature_map = torch.zeros((mask.shape[0], mask.shape[1], self.dim), dtype=torch.float32)
            feature_map[mask > 128] = self.text_embeddings[index]  # Assuming mask is binary with values 0 and 255
        return feature_map

    def extract_features(self, frame, metadata):
        file_name = metadata["image_name"].split("/")[-1]
        stem = os.path.splitext(file_name)[0]
        features_path = os.path.join(self.feature_dir, f"{stem}")
        if not os.path.exists(features_path):
            return None
        
        feats = self._create_feature_map(features_path)
        return feats