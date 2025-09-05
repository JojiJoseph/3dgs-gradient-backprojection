# This script is used for evaluating langsplat features

from collections import defaultdict
import json
import time
from typing import Literal
import glob
import tyro
import os
import torch
import cv2
import open_clip
import yaml
from dataclasses import dataclass, field

import numpy as np
import matplotlib
matplotlib.use("TkAgg")

from utils import (
    prune_by_gradients,
    test_proper_pruning,
    get_viewmat_from_colmap_image,
    load_checkpoint,
    torch_to_cv,
)
from utils import FeatureRenderer, to_builtin
import pprint


class LangSplatHelper:  # Not using CLIP/ClipModel to avoid confusion with clip_model
    def __init__(self):
        self.clip_model, self.preprocess_train, self.preprocess_val = (
            open_clip.create_model_and_transforms(
                "ViT-B-16", pretrained="laion2b_s34b_b88k"
            )
        )
        self.clip_model.to("cuda")
        neg_prompts = ["object", "things", "stuff", "texture"]
        neg_prompt_tokenized = open_clip.tokenize(neg_prompts).to("cuda")
        self.neg_prompt_embedding = self.clip_model.encode_text(
            neg_prompt_tokenized
        ).float()
        self.neg_prompt_embedding = torch.nn.functional.normalize(
            self.neg_prompt_embedding, dim=-1, eps=1e-5
        )

    def encode_text(self, text, normalize=False):
        embedding = self.clip_model.encode_text(
            open_clip.tokenize([text]).to("cuda")
        ).float()
        if normalize:
            embedding = torch.nn.functional.normalize(embedding, dim=-1, eps=1e-5)
        return embedding

    def get_relevancy_map(self, feats, prompt_embedding=None, inv_temp=10, smooth=True):
        if prompt_embedding.shape[0] == 1:
            prompt_embedding = prompt_embedding.repeat(
                len(self.neg_prompt_embedding), 1
            )
        sim_to_neg = feats @ self.neg_prompt_embedding.T  # (H, W, 4)
        sim_to_pos = feats @ prompt_embedding.T  # (H, W, 4)
        softmax = torch.softmax(
            inv_temp * torch.stack([sim_to_pos, sim_to_neg], dim=-1), dim=-1
        )
        score = softmax[..., 0].min(dim=-1).values
        if smooth:
            score_np = score.detach().cpu().numpy()
            score_np = cv2.blur(score_np, (20, 20))
            score = torch.from_numpy(score_np).to(score.device)
        max_score = score.max()
        mask = score
        return mask, max_score

    def get_location_and_relevancy_maps(self, feature_maps: list[torch.tensor], prompt: str):
        """
        Computes relevancy maps for a list of feature maps given a text prompt, and returns the location of the most relevant point, the combined relevancy map, and the most relevant individual map.

        Args:
            feature_maps (list[torch.Tensor]): A list of 2D feature maps (each as a torch.Tensor) to evaluate relevancy against the prompt.
            prompt (str): The text prompt to compute relevancy for.

        Returns:
            tuple:
                - (x, y) (tuple of int): The (x, y) coordinates of the maximum relevancy in the combined relevancy map.
                - combined_relevancy_map (np.ndarray): The combined relevancy map as a 2D numpy array, where each position contains the maximum relevancy score across all feature maps.
                - max_relevancy_map (np.ndarray): The relevancy map (as a 2D numpy array) from the feature map with the highest maximum relevancy score.

        Note:
            This method assumes that the feature maps and prompt embedding are compatible in terms of dimensions and device placement.
        """

        relevancy_maps = []
        prompt_embedding = self.encode_text(prompt, normalize=True).float()
        combined_relevancy_map = torch.zeros(
            (feature_maps[0].shape[0], feature_maps[0].shape[1])
        ).cuda()

        relevancy_scores = []

        for features in feature_maps:
            mask, max_score = self.get_relevancy_map(
                features, prompt_embedding=prompt_embedding
            )
            relevancy_maps.append(mask)
            relevancy_scores.append(max_score.item())
            combined_relevancy_map = torch.maximum(combined_relevancy_map, mask)

        combined_relevancy_map = combined_relevancy_map.detach().cpu().numpy()

        # Get x, y coordinate of maximum value
        y, x = np.unravel_index(np.argmax(combined_relevancy_map), combined_relevancy_map.shape)

        max_relevancy_map = relevancy_maps[np.argmax(relevancy_scores)].detach().cpu().numpy()

        return (x, y), combined_relevancy_map, max_relevancy_map


class LocalizationEvaluator:
    def __init__(self):
        self.counter = defaultdict(lambda: defaultdict(bool))

    def update(self, file, object, is_correct):
        self.counter[file][object] |= is_correct

    def get_stats(self):
        total_count = sum(len(self.counter[file]) for file in self.counter)
        total_correct = sum(
            self.counter[file][object]
            for file in self.counter
            for object in self.counter[file]
        )
        acc = total_correct / total_count if total_count > 0 else 0
        return total_correct, total_count, acc

    def print_stats(self):
        total_correct, total_count, acc = self.get_stats()
        print(f"Total Correct\t:\t{total_correct}")
        print(f"Total Count\t:\t{total_count}")
        print(f"Accuracy\t:\t{acc * 100:.2f}%")


class IoUEvaluator:
    def __init__(self):
        self.gt_masks = defaultdict(lambda: defaultdict(lambda: None))
        self.masks = defaultdict(lambda: defaultdict(lambda: None))

        self.mIoU = 0.0

    def update(self, frame, object, pred_mask, gt_mask):
        if self.gt_masks[frame][object] is None:
            self.gt_masks[frame][object] = gt_mask != 0
        else:
            self.gt_masks[frame][object][gt_mask != 0] = True
        self.masks[frame][object] = pred_mask != 0

    def _calc_iou(self, mask1, mask2):
        intersection = (mask1 & mask2).sum()
        union = (mask1 | mask2).sum()
        return intersection / union if union > 0 else 0

    def calc(self):
        count = 0
        iou_sum = 0
        for frame in self.gt_masks:
            for object in self.gt_masks[frame]:
                count += 1
                gt_mask = self.gt_masks[frame][object]
                pred_mask = self.masks[frame][object]
                if gt_mask is not None and pred_mask is not None:
                    iou_sum += self._calc_iou(gt_mask, pred_mask)
        self.mIoU = iou_sum / count if count > 0 else 0
        return self.mIoU

    def print_stats(self, recalc=True):
        if recalc:
            self.calc()
        print(f"mIoU\t:\t{self.mIoU * 100:.2f}%")








@dataclass
class Args:
    data_dir: str = "./data/lerf_ovs"  # Path to the COLMAP data directory.
    checkpoint: str = (
        "./data/lerf_ovs/teatime/ckpts/ckpt_29999_rank0.pt"  # Path to the checkpoint file, can be generated from the original 3DGS repo.
    )
    results_dir: str = "./results/lerf_ovs/teatime"  # Output directory for results.
    format: Literal["inria", "gsplat"] = (
        "gsplat"  # Checkpoint format: "inria" (original) or "gsplat".
    )
    data_factor: int = 1  # Downscale resolution by this factor.
    tag: str = None  # Optional tag for this evaluation run.
    prune: bool = True  # Whether to prune the 3DGS using gradients.
    feature_paths: list[str] = field(
        default_factory=lambda: [
            "./teatime_feats/teatime_level_1.pt",
            "./teatime_feats/teatime_level_2.pt",
            "./teatime_feats/teatime_level_3.pt",
        ]
    )  # Paths to the feature files for each level.
    label_dir: str = (
        "./data/lerf_ovs/label/teatime"  # Directory containing evaluation label files.
    )


def get_stem(file_name):
    return os.path.splitext(file_name)[0]


def is_inside_bbox(location, bbox):
    x, y = location
    return (x >= bbox[0]) and (x <= bbox[2]) and (y >= bbox[1]) and (y <= bbox[3])

def relevancy_map_to_mask(relevancy_map: np.ndarray, thresh: float = 0.4, eps: float = 1e-9):
    """
    Converts a relevancy map to a binary mask based on a threshold.

    The function first normalizes the input relevancy map to the range [0, 1], then rescales it to the range [-1, 1].
    It applies a threshold to generate a binary mask, where values above the threshold are set to 255 and others to 0.

    Args:
        relevancy_map (np.ndarray): Input relevancy map as a NumPy array.
        thresh (float, optional): Threshold value in the range [-1, 1] for mask generation. Defaults to 0.4.
        eps (float, optional): Small epsilon value to avoid division by zero. Defaults to 1e-9.

    Returns:
        np.ndarray: Binary mask as a uint8 NumPy array with values 0 or 255.
    """
    # Normalize relevancy map in the range 0 to 1
    relevancy_map = relevancy_map - relevancy_map.min()
    relevancy_map = relevancy_map / (relevancy_map.max() + eps)

    # Normalize relevancy_map to -1 to 1 and then threshold
    mask = relevancy_map * 2 - 1 > thresh

    # Convert it into uint8 mask
    return mask.astype(np.uint8) * 255


def main(args: Args):

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo")

    torch.set_default_device("cuda")

    # Already made inside __main__, but keeping in case any change in code outside the calling of this function
    os.makedirs(args.results_dir, exist_ok=True)
    
    splats = load_checkpoint(
        args.checkpoint, args.data_dir, format=args.format, data_factor=args.data_factor
    )

    if args.prune:
        splats_optimized = prune_by_gradients(splats)
        test_proper_pruning(splats, splats_optimized)
        splats = splats_optimized

    # Load features for all 3 levels used for evaluation
    feature_path_1, feature_path_2, feature_path_3 = args.feature_paths

    features_1 = torch.load(feature_path_1)
    features_2 = torch.load(feature_path_2)
    features_3 = torch.load(feature_path_3)

    # Get all the json files in this label directory
    label_files = glob.glob(os.path.join(args.label_dir, "*.json"))

    label_file_stems = [os.path.splitext(os.path.basename(f))[0] for f in label_files]
    label_file_stems.sort()

    loc_evaluator = LocalizationEvaluator()
    iou_evaluator = IoUEvaluator()
    lang_splat_helper = LangSplatHelper()

    # Extract Gaussian Parameters
    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors = torch.cat([colors_dc, colors_rest], dim=1)
    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]
    K = splats["camera_matrix"]
    width = K[0, 2] * 2
    height = K[1, 2] * 2

    renderer = FeatureRenderer(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=None,
        Ks=None,
        width=width,
        height=height,
    )

    colmap_val_images = filter(
        lambda x: get_stem(x.name) in label_file_stems,
        splats["colmap_project"].images.values(),
    )

    # Localization
    for image in sorted(colmap_val_images, key=lambda x: x.name):
        file_name = image.name
        stem = get_stem(file_name)
        json_path = os.path.join(args.label_dir, f"{stem}.json")
        
        with open(json_path) as f:
            annotations = json.load(f)

        viewmat = get_viewmat_from_colmap_image(image)
        feature_maps = []
        for features in [features_1, features_2, features_3]:

            features_rendered, _, _ = renderer.render_features(
                features=features,
                viewmats=viewmat[None],
                Ks=K[None],
                width=width,
                height=height,
            )
            features_rendered = features_rendered[0]
            features_rendered = torch.nn.functional.normalize(
                features_rendered, dim=-1, p=2
            )
            feature_maps.append(features_rendered)

        for object in annotations["objects"]:
            category = object["category"]
            segmentation = (
                np.array(object["segmentation"]).reshape(-1, 2).astype(np.int32)
            )
            gt_mask = np.zeros((int(height), int(width)), dtype=np.uint8)
            cv2.fillPoly(gt_mask, [segmentation], 255)

            location, combined_relevancy_map, max_relevancy_map = lang_splat_helper.get_location_and_relevancy_maps(
                feature_maps, prompt=category
            )

            mask = relevancy_map_to_mask(max_relevancy_map, thresh=0.4)

            iou_evaluator.update(stem, category, mask, gt_mask)

            inside_flag = is_inside_bbox(location, object["bbox"])

            loc_evaluator.update(json_path, category, inside_flag)

    loc_evaluator.print_stats()
    iou_evaluator.print_stats()
    results_path = os.path.join(args.results_dir, f"langsplat_evaluation_{args.tag}.json")
    _, _, acc = loc_evaluator.get_stats()

    mIoU = iou_evaluator.calc()

    os.makedirs(args.results_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"accuracy": acc, "mIoU": mIoU}, f)
    print("mIoU", mIoU * 100)


if __name__ == "__main__":

    cfg = tyro.cli(Args)

    print("Configuration:")
    pprint.pprint(to_builtin(cfg))
    print("\n")

    # Save config
    os.makedirs(cfg.results_dir, exist_ok=True)
    tag = cfg.tag if cfg.tag is not None else time.strftime("%Y%m%d_%H%M%S")
    cfg_out_path = os.path.join(cfg.results_dir, f"langsplat_evaluation_{tag}.yaml")
    with open(cfg_out_path, "w") as f:
        yaml.dump(to_builtin(cfg), f)

    # Run Evaluation
    main(cfg)
