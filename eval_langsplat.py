from collections import defaultdict
import json
import time
from typing import Literal
import glob
import tyro
import os
import torch
import cv2

import numpy as np
import matplotlib
from sklearn.decomposition import PCA
import open_clip
import yaml
from dataclasses import dataclass, field
from utils import FeatureRenderer, to_builtin


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


def get_location_and_mask(feature_maps, prompt, lang_splat_helper: LangSplatHelper):

    masks = []
    prompt_embedding = lang_splat_helper.encode_text(prompt, normalize=True).float()
    final_mask = torch.zeros(
        (feature_maps[0].shape[0], feature_maps[0].shape[1])
    ).cuda()

    relavancy_scores = []

    for features in feature_maps:
        mask, max_score = lang_splat_helper.get_relevancy_map(
            features, prompt_embedding=prompt_embedding
        )
        masks.append(mask)
        relavancy_scores.append(max_score.item())
        final_mask = torch.maximum(final_mask, mask)

    final_mask = final_mask.detach().cpu().numpy()

    # Get x, y coordinate of maximum value
    y, x = np.unravel_index(np.argmax(final_mask), final_mask.shape)

    seg_mask = masks[np.argmax(relavancy_scores)].detach().cpu().numpy()

    return (x, y), final_mask, seg_mask


matplotlib.use("TkAgg")

from utils import (
    prune_by_gradients,
    test_proper_pruning,
    get_viewmat_from_colmap_image,
    load_checkpoint,
    torch_to_cv,
)


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
    show_visual_feedback: bool = (
        True  # Whether to show visual feedback during evaluation.
    )
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


def main(args: Args):

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo")

    torch.set_default_device("cuda")

    os.makedirs(args.results_dir, exist_ok=True)
    splats = load_checkpoint(
        args.checkpoint, args.data_dir, format=args.format, data_factor=args.data_factor
    )
    if args.prune:
        splats_optimized = prune_by_gradients(splats)
        test_proper_pruning(splats, splats_optimized)
        splats = splats_optimized

    # Load all 4 checkpoints
    feature_path_1, feature_path_2, feature_path_3 = args.feature_paths

    features_1 = torch.load(feature_path_1)
    features_2 = torch.load(feature_path_2)
    features_3 = torch.load(feature_path_3)

    # Get all the json files in this label directory
    json_files = glob.glob(os.path.join(args.label_dir, "*.json"))

    json_file_stems = [os.path.splitext(os.path.basename(f))[0] for f in json_files]
    json_file_stems.sort()

    loc_evaluator = LocalizationEvaluator()
    iou_evaluator = IoUEvaluator()

    # if False:
    #     render_pca(
    #         splats,
    #         features_1,
    #         None,  # f"{results_dir}/pca_gaussians_{tag}.gif",
    #         pca_on_gaussians=True,
    #         scale=1.0,
    #         feedback=show_visual_feedback,
    #     )
    #     render_pca(
    #         splats,
    #         features_2,
    #         None,  # f"{results_dir}/pca_gaussians_{tag}.gif",
    #         pca_on_gaussians=True,
    #         scale=1.0,
    #         feedback=show_visual_feedback,
    #     )
    #     render_pca(
    #         splats,
    #         features_3,
    #         None,  # f"{results_dir}/pca_gaussians_{tag}.gif",
    #         pca_on_gaussians=True,
    #         scale=1.0,
    #         feedback=show_visual_feedback,
    #     )

    lang_splat_helper = LangSplatHelper()

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
        width=None,
        height=None,
    )

    colmap_val_images = filter(
        lambda x: get_stem(x.name) in json_file_stems,
        splats["colmap_project"].images.values(),
    )

    # Localization
    for image in sorted(colmap_val_images, key=lambda x: x.name):
        file_name = image.name
        stem = get_stem(file_name)
        json_path = os.path.join(args.label_dir, f"{stem}.json")
        annotations = json.load(open(json_path))

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

            location, mask_combined, mask_max_relevancy = get_location_and_mask(
                feature_maps, prompt=category, lang_splat_helper=lang_splat_helper
            )

            mask_thresh = mask_max_relevancy - mask_max_relevancy.min()
            mask_thresh = mask_thresh / (mask_thresh.max() + 1e-5)

            mask = mask_thresh * 2 - 1 > 0.4
            mask = mask.astype(np.uint8) * 255

            iou_evaluator.update(stem, category, mask, gt_mask)

            inside_flag = is_inside_bbox(location, object["bbox"])

            loc_evaluator.update(json_path, category, inside_flag)

    loc_evaluator.print_stats()
    iou_evaluator.print_stats()
    results_path = os.path.join(cfg.results_dir, f"langsplat_evaluation_{cfg.tag}.json")
    _, _, acc = loc_evaluator.get_stats()

    mIoU = iou_evaluator.calc()

    os.makedirs(cfg.results_dir, exist_ok=True)
    json.dump({"accuracy": acc, "mIoU": mIoU}, open(results_path, "w"))
    print("mIoU", mIoU * 100)


if __name__ == "__main__":

    cfg = tyro.cli(Args)

    print("Configuration:")
    print(cfg)
    print("\n")

    # Save config
    os.makedirs(cfg.results_dir, exist_ok=True)
    cfg_out_path = os.path.join(cfg.results_dir, f"langsplat_evaluation_{cfg.tag}.yaml")
    with open(cfg_out_path, "w") as f:
        yaml.dump(to_builtin(cfg), f)

    # Run Evaluation
    main(cfg)
