# This script is used for evaluating langsplat features

from collections import defaultdict
import json
import time
from typing import Literal, Union
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

matplotlib.use("TkAgg") # To avoid conflict with opencv

from utils import (
    prune_by_gradients,
    test_proper_pruning,
    get_viewmat_from_colmap_image,
    load_checkpoint,
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

    def encode_text(
        self, text: str, normalize: bool = False, eps: float = 1e-9
    ) -> torch.Tensor:
        """
        Encodes the input text into a feature embedding using the CLIP model.

        Args:
            text (str): The input text string to encode.
            normalize (bool, optional): If True, the resulting embedding is L2-normalized. Defaults to False.
            eps (float, optional): A small value added for numerical stability during normalization. Defaults to 1e-9.

        Returns:
            torch.Tensor: The encoded text embedding as a float tensor.
        """
        embedding = self.clip_model.encode_text(
            open_clip.tokenize([text]).to("cuda")
        ).float()
        if normalize:
            embedding = torch.nn.functional.normalize(embedding, dim=-1, eps=eps)
        return embedding

    def get_relevancy_map(
        self,
        feats: torch.Tensor,
        prompt_embedding: torch.Tensor = None,
        inv_temp: Union[float, int] = 10,
        smooth: bool = True,
        smooth_kernel_size: int = 20,
    ):
        """
        Computes a relevancy map based on feature similarities to positive and negative prompt embeddings.

        Args:
            feats (torch.Tensor): Feature tensor of shape (H, W, D), where D is the feature dimension.
            prompt_embedding (torch.Tensor, optional): Positive prompt embedding tensor of shape (N, D).
                If shape is (1, D), it is repeated to match the number of negative prompt embeddings.
            inv_temp (Union[float, int], optional): Inverse temperature scaling factor for softmax. Default is 10.
            smooth (bool, optional): Whether to apply smoothing (blurring) to the relevancy map. Default is True.
            smooth_kernel_size (int, optional): The kernel size to use for smoothing. Default is 20.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - relevancy_map (torch.Tensor): The computed relevancy map of shape (H, W).
                - max_score (torch.Tensor): The maximum relevancy score in the map.
        """
        if prompt_embedding.shape[0] == 1:
            # Match the shape same as negative prompt embeddings
            prompt_embedding = prompt_embedding.repeat(
                len(self.neg_prompt_embedding), 1
            )

        # Calculate similarity of feats with both negative and positive embeddings
        sim_to_neg = feats @ self.neg_prompt_embedding.T  # (H, W, 4)
        sim_to_pos = feats @ prompt_embedding.T  # (H, W, 4)

        # Convert feature similarities to class probabilities: [positive, negative] in last dimension
        softmax = torch.softmax(
            inv_temp * torch.stack([sim_to_pos, sim_to_neg], dim=-1), dim=-1
        ) # (H, W, 4, 2)

        # Relevancy map represents minimum of probabilities of being positive
        relevancy_map = softmax[..., 0].min(dim=-1).values
        if smooth:
            # Optionally smoothes out the relevancy map
            relevancy_map_np = relevancy_map.detach().cpu().numpy()
            relevancy_map_np = cv2.blur(relevancy_map_np, (smooth_kernel_size, smooth_kernel_size))
            relevancy_map = torch.from_numpy(relevancy_map_np).to(relevancy_map.device)
        max_score = relevancy_map.max()
        return relevancy_map, max_score

    def get_location_and_relevancy_maps(
        self, feature_maps: list[torch.Tensor], prompt: str
    ):
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

        relevancy_map_list = []
        prompt_embedding = self.encode_text(prompt, normalize=True).float()
        combined_relevancy_map = torch.zeros(
            (feature_maps[0].shape[0], feature_maps[0].shape[1])
        ).cuda()

        relevancy_scores = []

        for features in feature_maps:
            relevancy_map, max_score = self.get_relevancy_map(
                features, prompt_embedding=prompt_embedding
            )
            relevancy_map_list.append(relevancy_map)
            relevancy_scores.append(max_score.item())
            combined_relevancy_map = torch.maximum(
                combined_relevancy_map, relevancy_map
            )

        combined_relevancy_map = combined_relevancy_map.detach().cpu().numpy()

        # Get x, y coordinate of maximum value
        y, x = np.unravel_index(
            np.argmax(combined_relevancy_map), combined_relevancy_map.shape
        )

        max_relevancy_map = (
            relevancy_map_list[np.argmax(relevancy_scores)].detach().cpu().numpy()
        )

        return (x, y), combined_relevancy_map, max_relevancy_map


class LocalizationEvaluator:
    """
    A class to evaluate localization accuracy by tracking correctness of object predictions per file.

    Attributes:
        counter (defaultdict): Nested dictionary mapping file names to objects and their correctness.

    Methods:
        update(file, object_, is_correct):
            Updates the correctness status for a given object in a file.

        get_stats():
            Computes and returns the total number of correct predictions, total predictions, and accuracy.

        print_stats():
            Prints the total correct predictions, total predictions, and accuracy in a formatted manner.
    """
    def __init__(self):
        # counter is a nested dictionary structured as:
        # {
        #     file: {
        #         object: is_localized
        #     }
        # }
        # Note: in a file multiple instances of same object can exist
        # But prompting with LangSplat features can localize only one object.
        # Therefore if we locate any instance of that object we consider it as localized.
        self.counter = defaultdict(lambda: defaultdict(bool))

    def update(self, file: str, object_: str, is_localized: bool):
        """
        Updates the localization status of an object within a file.

        If at least one instance of the object is localized in the file, the corresponding entry in the counter is marked as localized.

        Args:
            file (str): The name or path of the file being evaluated.
            object_ (str): The identifier of the object within the file.
            is_localized (bool): Whether the object is localized in this instance.

        """
        # The counter[file][object_] is marked localized if at least one instance of the object is localized in the file.
        self.counter[file][object_] |= is_localized

    def get_stats(self):
        """
        Calculates and returns statistics based on the internal counter.

        Returns:
            tuple: A tuple containing:
                - acc (float): The accuracy, defined as total_localized divided by total_count (0 if total_count is 0).
                - total_localized (int): The total count of localized objects across all files.
                - total_count (int): The total number of objects across all files.
        """
        total_count = sum(len(self.counter[file]) for file in self.counter)
        total_localized = sum(
            self.counter[file][object_]
            for file in self.counter
            for object_ in self.counter[file]
        )
        acc = total_localized / total_count if total_count > 0 else 0
        return acc, total_localized, total_count

    def print_stats(self):
        """
        Prints statistics about localization performance.

        Retrieves accuracy, total localized items, and total count using `get_stats()`,
        then prints these values in a formatted manner:
            - Total Localized: Number of items successfully localized.
            - Total Count: Total number of items evaluated.
            - Accuracy: Percentage of items correctly localized.

        Returns:
            None
        """
        acc, total_localized, total_count = self.get_stats()
        print(f"Total Localized\t:\t{total_localized}")
        print(f"Total Count\t:\t{total_count}")
        print(f"Accuracy\t:\t{acc * 100:.2f}%")


class IoUEvaluator:
    """
    Evaluator for computing mean Intersection over Union (mIoU) between predicted and ground truth masks.
    This class manages nested dictionaries of masks for multiple frames and objects, allowing incremental updates and efficient mIoU calculation.
    Attributes:
        gt_masks (defaultdict): Nested dictionary storing ground truth masks for each frame and object.
        masks (defaultdict): Nested dictionary storing predicted masks for each frame and object.
        mIoU (float): The mean Intersection over Union score.
    Methods:
        update(frame, object_, pred_mask, gt_mask):
            Updates the evaluator with a new predicted and ground truth mask for a given frame and object.
        _calc_iou(mask1, mask2):
            Computes the Intersection over Union (IoU) between two binary masks.
        calc():
            Calculates and returns the mean IoU (mIoU) over all stored masks.
        print_stats(recalc=True):
            Prints the current mIoU score. Optionally recalculates before printing.
    """
    def __init__(self):
        # gt_masks and masks are nested dictionaries structured as:
        # {
        #     frame: {
        #         object: 2D_mask_array
        #     }
        # }
        #   - The first level key is the frame (file name of the frame).
        #   - The second level key is the object (the text/prompt for the object).
        #   - The leaf nodes are 2D masks
        self.gt_masks = defaultdict(lambda: defaultdict(lambda: None))
        self.masks = defaultdict(lambda: defaultdict(lambda: None))
        self.mIoU = 0.0

    def update(
        self,
        frame: str,
        object_: str,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray
    ) -> None:
        """
        Updates the evaluator with predicted and ground truth masks for a specific frame and object.

        If a ground truth mask for the given frame and object does not exist, it is initialized.
        Otherwise, the new ground truth mask is merged with the existing one.
        The predicted mask is always updated for the given frame and object.

        Args:
            frame (str): The frame identifier (e.g., file name).
            object_ (str): The object identifier (e.g., text prompt).
            pred_mask (np.ndarray): The predicted binary mask for the object.
            gt_mask (np.ndarray): The ground truth binary mask for the object.
        """
        if self.gt_masks[frame][object_] is None:
            # If no ground truth mask exists, create one
            self.gt_masks[frame][object_] = gt_mask != 0
        else:
            # Merge mask if there is already one present
            self.gt_masks[frame][object_][gt_mask != 0] = True
        # We need not check if object_ already exists in current frame,
        # Because one frame with one prompt creates only one mask
        self.masks[frame][object_] = pred_mask != 0

    def _calc_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Computes the Intersection over Union (IoU) between two binary masks.

        Args:
            mask1 (np.ndarray): First binary mask (boolean).
            mask2 (np.ndarray): Second binary mask (boolean).

        Returns:
            float: The IoU score between the two masks.
        """
        # Ensure masks are boolean arrays
        intersection = (mask1 & mask2).sum()
        union = (mask1 | mask2).sum()
        return intersection / union if union > 0 else 0

    def calc(self) -> float:
        """
        Calculates the mean Intersection over Union (mIoU) between ground truth masks and predicted masks.

        Iterates over all frames and objects, computes the IoU for each valid ground truth and predicted mask pair,
        and returns the average IoU across all pairs.

        Returns:
            float: The mean IoU (mIoU) value. Returns 0 if there are no valid mask pairs.
        """
        count = 0
        iou_sum = 0
        for frame in self.gt_masks:
            for object_ in self.gt_masks[frame]:
                count += 1
                gt_mask = self.gt_masks[frame][object_]
                pred_mask = self.masks[frame][object_]
                if gt_mask is not None and pred_mask is not None:
                    iou_sum += self._calc_iou(gt_mask, pred_mask)
        self.mIoU = iou_sum / count if count > 0 else 0
        return self.mIoU

    def print_stats(self, recalc: bool = True):
        """
        Prints the mean Intersection over Union (mIoU) statistics.

        Args:
            recalc (bool, optional): If True, recalculates the statistics before printing. Defaults to True.

        Side Effects:
            Prints the mIoU value as a percentage to the standard output.
        """
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


def get_stem(file_name: str) -> str:
    """
    Returns the stem (file name without extension).
    """
    return os.path.splitext(file_name)[0]


def is_inside_bbox(location, bbox) -> bool:
    """
    Check if a 2D point is inside a given bounding box.

    Args:
        location (array-like): (x, y) coordinates of the point.
        bbox (array-like): (x_min, y_min, x_max, y_max) bounding box.

    Returns:
        bool: True if the point is inside or on the edge of the bounding box, False otherwise.
    """
    x, y = location
    return (x >= bbox[0]) and (x <= bbox[2]) and (y >= bbox[1]) and (y <= bbox[3])


def relevancy_map_to_mask(
    relevancy_map: np.ndarray, thresh: float = 0.4, eps: float = 1e-9
):
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

            location, combined_relevancy_map, max_relevancy_map = (
                lang_splat_helper.get_location_and_relevancy_maps(
                    feature_maps, prompt=category
                )
            )

            mask = relevancy_map_to_mask(max_relevancy_map, thresh=0.4)

            iou_evaluator.update(stem, category, mask, gt_mask)

            inside_flag = is_inside_bbox(location, object["bbox"])

            loc_evaluator.update(json_path, category, inside_flag)

    # Print evaluation results
    loc_evaluator.print_stats()
    iou_evaluator.print_stats()
    acc, _, _ = loc_evaluator.get_stats()
    mIoU = iou_evaluator.calc()

    
    # Save the results
    results_path = os.path.join(
        args.results_dir, f"langsplat_evaluation_{args.tag}.json"
    )
    os.makedirs(args.results_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"accuracy": acc, "mIoU": mIoU}, f)



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
