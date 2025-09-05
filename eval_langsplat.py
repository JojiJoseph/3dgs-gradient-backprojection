from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any, Literal
import glob
from enum import Enum
import tyro
import os
import torch
import cv2
import imageio  # To generate gifs
import pycolmap_scene_manager as pycolmap
from gsplat import rasterization
import numpy as np
import clip
import matplotlib
from sklearn.decomposition import PCA
import open_clip
import yaml
from utils import FeatureRenderer, to_builtin


clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    "ViT-B-16", pretrained="laion2b_s34b_b88k"
)
clip_model.to("cuda")
prompt = "three coockies"
prompt_tokenized = open_clip.tokenize([prompt]).to("cuda")
prompt_embedding = clip_model.encode_text(prompt_tokenized).float()

neg_prompts = ["object", "things", "stuff", "texture"]
neg_prompt_tokenized = open_clip.tokenize(neg_prompts).to("cuda")
neg_prompt_embedding = clip_model.encode_text(neg_prompt_tokenized).float()

pos_prompt = prompt_embedding.repeat(len(neg_prompts), 1)

# Normalize prompts
# pos_prompt_embedding = pos_prompt / pos_prompt.norm(dim=-1, keepdim=True)
# neg_prompt_embedding = neg_prompt_embedding / neg_prompt_embedding.norm(dim=-1, keepdim=True)
pos_prompt = torch.nn.functional.normalize(pos_prompt, dim=-1, eps=1e-5)
neg_prompt_embedding = torch.nn.functional.normalize(
    neg_prompt_embedding, dim=-1, eps=1e-5
)
# print(pos_prompt_embedding.shape, neg_prompt_embedding.shape)
# exit()
# Both has shape (4,512), Stack them like (4,2,512) and take softmax over dim 1
stacked = torch.stack([pos_prompt, neg_prompt_embedding], dim=1)


# sims = torch.softmax(10*stacked, dim=1)
# score_ = sims[:,0]
# print(stacked.shape, sims.shape)
# exit()
def get_mask(feats, pos_prompt=pos_prompt, inv_temp=10, thresh=0.5, smooth=True):
    # print(pos_prompt.shape, neg_prompt_embedding.shape)
    # exit()
    if pos_prompt.shape[0] == 1:
        pos_prompt = pos_prompt.repeat(len(neg_prompt_embedding), 1)
    sim_to_neg = feats @ neg_prompt_embedding.T  # (H, W, 4)
    sim_to_pos = feats @ pos_prompt.T  # (H, W, 4)
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


def get_location_and_mask(feature_maps, prompt):

    masks = []
    prompt_embedding = clip_model.encode_text(
        open_clip.tokenize([prompt]).to("cuda")
    ).float()
    # Normalize prompt_embedding
    prompt_embedding = torch.nn.functional.normalize(prompt_embedding, dim=-1, eps=1e-5)
    final_mask = torch.zeros(
        (feature_maps[0].shape[0], feature_maps[0].shape[1])
    ).cuda()

    relavancy_scores = []

    for features in feature_maps:
        # Feature is of shape (H,W,512), apply kernel on it

        # assert features_smoothed.shape == features.shape, (features_smoothed.shape, features.shape)
        mask, max_score = get_mask(features, pos_prompt=prompt_embedding)
        masks.append(mask)
        relavancy_scores.append(max_score.item())
        final_mask = torch.maximum(final_mask, mask)

    # final_mask = torch.maximum(*masks).detach().cpu().numpy()
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


def render_pca(
    splats,
    features,
    output_path,
    pca_on_gaussians=True,
    scale=1.0,
    feedback=True,
):
    if feedback:
        cv2.destroyAllWindows()
        cv2.namedWindow("PCA", cv2.WINDOW_NORMAL)
    frames = []
    means = splats["means"]
    colors_dc = splats["features_dc"]
    colors_rest = splats["features_rest"]
    colors = torch.cat([colors_dc, colors_rest], dim=1)
    opacities = torch.sigmoid(splats["opacity"])
    scales = torch.exp(splats["scaling"])
    quats = splats["rotation"]
    K = splats["camera_matrix"]
    if output_path is not None:
        aux_dir = output_path + ".images"
        os.makedirs(aux_dir, exist_ok=True)

    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features.detach().cpu().numpy())
    feats_min = np.min(features_pca, axis=(0, 1))
    feats_max = np.max(features_pca, axis=(0, 1))
    features_pca = (features_pca - feats_min) / (feats_max - feats_min)
    features_pca = torch.tensor(features_pca).float().cuda()
    if pca_on_gaussians:
        for image in sorted(
            splats["colmap_project"].images.values(), key=lambda x: x.name
        ):
            viewmat = get_viewmat_from_colmap_image(image)
            features_rendered, alphas, meta = rasterization(
                means,
                quats,
                scales * scale,
                opacities,
                features_pca,
                viewmats=viewmat[None],
                Ks=K[None],
                width=K[0, 2] * 2,
                height=K[1, 2] * 2,
                # sh_degree=3,
            )
            features_rendered = features_rendered[0]
            frame = torch_to_cv(features_rendered)
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            frames.append(frame)
            if feedback:
                cv2.imshow("PCA", frame[..., ::-1])
                if output_path is not None:
                    cv2.imwrite(f"{aux_dir}/{image.name}", frame[..., ::-1])
                cv2.waitKey(1)
    else:
        for image in sorted(
            splats["colmap_project"].images.values(), key=lambda x: x.name
        ):
            viewmat = get_viewmat_from_colmap_image(image)
            features_rendered, alphas, meta = rasterization(
                means,
                quats,
                scales * scale,
                opacities,
                features,
                viewmats=viewmat[None],
                Ks=K[None],
                width=K[0, 2] * 2,
                height=K[1, 2] * 2,
                # sh_degree=3,
            )
            features_rendered = features_rendered[0]
            h, w, c = features_rendered.shape
            features_rendered = (
                features_rendered.reshape(h * w, c).detach().cpu().numpy()
            )
            features_rendered = pca.transform(features_rendered)
            features_rendered = features_rendered.reshape(h, w, 3)
            features_rendered = (features_rendered - feats_min) / (
                feats_max - feats_min
            )
            frame = (features_rendered * 255).astype(np.uint8)
            frames.append(frame[..., ::-1])
            if feedback:
                cv2.imshow("PCA", frame)
                if output_path is not None:
                    cv2.imwrite(f"{aux_dir}/{image.name}", frame)
                cv2.waitKey(1)
    if output_path is not None:
        imageio.mimsave(output_path, frames, fps=10, loop=0)
    if feedback:
        cv2.destroyAllWindows()


from dataclasses import dataclass, field, is_dataclass


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

    if False:
        render_pca(
            splats,
            features_1,
            None,  # f"{results_dir}/pca_gaussians_{tag}.gif",
            pca_on_gaussians=True,
            scale=1.0,
            feedback=show_visual_feedback,
        )
        render_pca(
            splats,
            features_2,
            None,  # f"{results_dir}/pca_gaussians_{tag}.gif",
            pca_on_gaussians=True,
            scale=1.0,
            feedback=show_visual_feedback,
        )
        render_pca(
            splats,
            features_3,
            None,  # f"{results_dir}/pca_gaussians_{tag}.gif",
            pca_on_gaussians=True,
            scale=1.0,
            feedback=show_visual_feedback,
        )

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
                feature_maps, prompt=category
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
