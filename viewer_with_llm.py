# Basic OpenCV viewer with sliders for rotation and translation.
# Can be easily customizable to different use cases.
from dataclasses import dataclass
from typing import Literal, Optional
import torch
from gsplat import rasterization
import cv2
import clip
from scipy.spatial.transform import Rotation as scipyR
import pycolmap_scene_manager as pycolmap
import warnings
from torchvision.transforms import functional as TF

import numpy as np
import json
import tyro
from lseg import LSegNet

from utils import (
    get_rpy_matrix,
    get_viewmat_from_colmap_image,
    prune_by_gradients,
    torch_to_cv,
    load_checkpoint,
)

# Check if CUDA is available. Else raise an error.
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. Please install the correct version of PyTorch with CUDA support."
    )

device = torch.device("cuda")
torch.set_default_device("cuda")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from transformers import pipeline



from typing import List, Dict, Optional
import json
import warnings

def get_mask3d_lseg(splats, features, prompt, neg_prompt, threshold=None):

    net = LSegNet(
        backbone="clip_vitl16_384",
        features=256,
        crop_size=480,
        arch_option=0,
        block_depth=0,
        activation="lrelu",
    )
    # Load pre-trained weights
    net.load_state_dict(torch.load("./checkpoints/lseg_minimal_e200.ckpt"))
    net.eval()
    net.cuda()

    # Preprocess the text prompt
    clip_text_encoder = net.clip_pretrained.encode_text

    positive_prompts_length = len(prompt.split(";"))

    prompts = prompt.split(";") + neg_prompt.split(";")

    prompts = clip.tokenize(prompts)
    prompts = prompts.cuda()

    text_feat = clip_text_encoder(prompts)  # N, 512, N - number of prompts
    text_feat_norm = torch.nn.functional.normalize(text_feat, dim=1)

    features = torch.nn.functional.normalize(features, dim=1)
    score = features @ text_feat_norm.float().T
    mask_3d = score[:, :positive_prompts_length].max(dim=1)[0] > score[:, positive_prompts_length:].max(dim=1)[0]
    if threshold is not None:
        mask_3d = mask_3d & (score[:, 0] > threshold)
    mask_3d_inv = ~mask_3d

    return mask_3d, mask_3d_inv

COLOR_TO_RGB = {
    "red": [1.0, 0.0, 0.0],
    "green": [0.0, 1.0, 0.0],
    "blue": [0.0, 0.0, 1.0],
    "yellow": [1.0, 1.0, 0.0],
    "cyan": [0.0, 1.0, 1.0],
    "magenta": [1.0, 0.0, 1.0],
    "white": [1.0, 1.0, 1.0],
    "black": [0.0, 0.0, 0.0],
}

class Assistant:
    def __init__(self):
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
        self.model = model
        self.tokenizer = tokenizer

        # Emulating system prompt with a series of example interactions
        self.system_prompt = [
                {
                    "role": "user",
                    "content": """
                        You are a 3DGS viewer assistant. You understand commands like changing the view (top, front, right),
                        segment 3d gaussians, change color, and exiting the application.
                        Always respond in strict JSON format: {"request": "<type>", "side": "<side>"} or {"request": "exit"}.
                        If the command is unclear, respond with {"request": "unknown"}.
                    """,
                },
                {
                    "role": "assistant",
                    "content": """
                    Ok!
                    """,
                },
                {
                    "role": "user",
                    "content": """
                    Can you show me the front view?
                    """,
                },
                {
                    "role": "assistant",
                    "content": """
                    {"request": "change_view", "side": "front"}
                    """,
                },
                {
                    "role": "user",
                    "content": """
                    Can you show me the right view?
                    """,
                },
                {
                    "role": "assistant",
                    "content": """
                    {"request": "change_view", "side": "right"}
                    """
                },
                {
                    "role": "user",
                    "content": """
                    Please change view to the top view.
                    """,
                },
                {
                    "role": "assistant",
                    "content": """
                    {"request": "change_view", "side": "top"}
                    """,
                },
                {
                    "role": "user",
                    "content": """
                    Abracadabra
                    """,
                },
                {
                    "role": "assistant",
                    "content": """
                    {"request": "unknown"}
                    """,
                },
                {
                    "role": "user",
                    "content": """
                    Who are you?
                    """,
                },
                {
                    "role": "assistant",
                    "content": """
                    {"request": "unknown"}
                    """,
                },
                {
                    "role": "user",
                    "content": """
                    Can you exit?
                    """,
                },
                {
                    "role": "assistant",
                    "content": """
                    {"request": "exit", "message": "Goodbye!"}
                    """,
                },
                {
                    "role": "user",
                    "content": """
                    Bye, please quit.
                    """,
                },
                {
                    "role": "assistant",
                    "content": """
                    {"request": "exit", "message": "Goodbye!"}
                    """,
                },
                {
                    "role": "user",
                    "content": """
                    Can you segment 3D Gaussians with the prompt "car" and negative prompt "tree"?
                    """,
                },
                {
                    "role": "assistant",
                    "content": """
                    {"request": "segment", "prompt": "car", "neg_prompt": "tree"}
                    """,
                },
                {
                    "role": "user",
                    "content": """
                    Can you segment 3D Gaussians with the prompt "table"?
                    """,
                },
                {
                    "role": "assistant",
                    "content": """
                    {"request": "segment", "prompt": "table", "neg_prompt": "none"}
                    """,
                },
                {
                    "role": "user",
                    "content": """
                    Can you segment 3D Gaussians containing table and exclude plant?
                    """,
                },
                {
                    "role": "assistant",
                    "content": """
                    {"request": "segment", "prompt": "table", "neg_prompt": "plant"}
                    """,
                },
                {
                    "role": "user",
                    "content": """Can you reset segmentation?""",
                },
                {
                    "role": "assistant",
                    "content": """
                    {"request": "reset_segmentation"}
                    """,
                },
                {
                    "role": "user",
                    "content": "Change the color of grass to red.",
                }, {
                    "role": "assistant",
                    "content": """
                    {"request": "change_color", "object": "grass", "color": "red"}
                    """,
                },
                {
                    "role": "user",
                    "content": "Change the color of table to blue.",
                }, {
                    "role": "assistant",
                    "content": """
                    {"request": "change_color", "object": "table", "color": "blue"}
                    """,
                }, {
                    "role": "user",
                    "content": "Reset the color of all objects.",
                }, {
                    "role": "assistant",
                    "content": """
                    {"request": "reset_color"}
                    """,
                }
            ]
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",)

    def ask(self, query: str, max_new_tokens: int = 512, temperature: float = 0.7) -> Optional[Dict]:
        if query.startswith("`"):
            query = query[1:]
        output = self.pipeline(self.system_prompt + [{'role': 'user', 'content': query}],
            max_new_tokens=200,
            do_sample=True,
            temperature=temperature,
        )
        response = output[0]['generated_text'][-1]["content"].strip()
        # Try to extract JSON from response
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            response_str = response[json_start:json_end]
            parsed_response = json.loads(response_str)
            return parsed_response
        except json.JSONDecodeError as e:
            warnings.warn(f"Failed to parse JSON from response: {e}\nResponse:\n{response}")
            return None
        
    def __call__(self, query: str, max_new_tokens: int = 512, temperature: float = 0.7) -> Optional[Dict]:
        """
        Call the assistant with a query.

        Args:
            query: User's question or command.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Parsed JSON response as dict, or None if parsing fails.
        """
        return self.ask(query, max_new_tokens, temperature)


@dataclass
class Args:
    checkpoint: str  # Path to the 3DGS checkpoint file (.pth/.pt) to be visualized.
    data_dir: (
        str  # Path to the COLMAP project directory containing sparse reconstruction.
    )
    format: Optional[Literal["inria", "gsplat", "ply"]] = (
        "gsplat"  # Format of the checkpoint: 'inria' (original 3DGS), 'gsplat', or 'ply'.
    )
    rasterizer: Optional[Literal["inria", "gsplat"]] = (
        None  # [Deprecated] Use --format instead. Provided for backward compatibility.
    )
    data_factor: int = 4  # Downscaling factor for the renderings.
    turntable: bool = True  # Whether to use a turntable mode for the viewer.
    lseg_checkpoint: Optional[str] = None  # Path to the LSEG checkpoint for segmentation, if available.


@dataclass
class ViewerArgs:
    turntable: bool = False

from viewer import Viewer

class ViewerWithAssistant(Viewer):
    def __init__(self, splats, viewer_args):
        super().__init__(splats, viewer_args)
        self.assistant = Assistant()
        self.current_view = None
        self.assistant_mode = False
        self.user_query = ""
        self.mask3d = None
        print(splats.keys())
        self.opacities_backup = torch.sigmoid(splats["opacity"].clone().detach())
        self.colors_backup = self.colors.clone().detach()
    def handle_key_press(self, key, data):
        if key == ord("`"):
            self.assistant_mode = True
        if not self.assistant_mode:
            should_continue = super().handle_key_press(key, data)
            return should_continue
        # if key == ord("q") or key == 27:
        #     return False
        # Check if the key is backspace
        if key == ord("\b") or key == 8:  # Backspace key
            if len(self.user_query) > 1:
                # Remove the last character from the assistant text
                self.user_query = self.user_query[:-1]
        # Check if the key is a printable character
        if 32 <= key <= 126:
            # Convert the key to a character and append it to the assistant text
            self.user_query += chr(key)
        elif key == ord("\n") or key == ord("\r"):
            # Process the assistant text when Enter is pressed
            json_response = self.assistant(self.user_query)
            print("json_response:", json_response)
            if json_response is None:
                warnings.warn(
                    f"Failed to parse assistant response as JSON: {self.user_query}"
                )
                self.user_query = ""
                return True
            if "request" not in json_response:
                warnings.warn(
                    f"Invalid assistant response: {json_response}. Expected 'request' key."
                )
                self.user_query = ""
                return True
            if json_response["request"] == "change_view":
                side = json_response.get("side", "top")
                viewmat = self._get_viewmat_from_trackbars()
                self.current_view = self.get_special_viewmat(viewmat, side)
                self.update_trackbars_from_viewmat(self.current_view)
            if json_response["request"] == "exit":
                print(json_response.get("message", "Exiting..."))
                return False
            if json_response["request"] == "segment":
                # reset opacities
                self.opacities = self.opacities_backup.clone().detach()
                prompt = json_response.get("prompt", "")
                neg_prompt = json_response.get("neg_prompt", "none")
                if not prompt:
                    warnings.warn("No prompt provided for segmentation.")
                if neg_prompt == "" or neg_prompt == "none":
                    neg_prompt = "other"
                else:
                    neg_prompt = neg_prompt + ";other"
                features = self.splats["lseg"]
                mask3d, mask3d_inv = get_mask3d_lseg(
                    self.splats,
                    features,
                    prompt,
                    neg_prompt,
                    # threshold=0.5,
                )
                self.opacities[~mask3d] = 0.0
            if json_response["request"] == "reset_segmentation":
                # reset opacities
                self.opacities = self.opacities_backup.clone().detach()
                self.mask3d = None
                print("Segmentation reset.")
            if json_response["request"] == "change_color":
                object_name = json_response.get("object", "")
                color = json_response.get("color", "white")
                if color not in COLOR_TO_RGB:
                    warnings.warn(f"Color '{color}' is not recognized. Please change COLOR_TO_RGB dictionary.")
                else:
                    features = self.splats["lseg"]
                    mask3d, mask3d_inv = get_mask3d_lseg(
                        self.splats,
                        features,
                        prompt=object_name,
                        neg_prompt="other",
                        # threshold=0.5,
                    )
                    colors = self.colors_backup[mask3d, 0, :].clone().detach() * 0.2820947917738781 + 0.5
                    grays = TF.rgb_to_grayscale(colors.permute(1,0).reshape(1,3,-1,1))[0,0]
                    self.colors[mask3d,0,:] = (torch.tensor(COLOR_TO_RGB[color], device=device)*grays[:,0:1] - 0.5) / 0.2820947917738781
            if json_response["request"] == "reset_color":
                self.mask3d = None
                self.colors = self.colors_backup.clone().detach()
            if json_response["request"] == "unknown":
                warnings.warn(
                    f"Assistant response is unknown: {json_response}. Please try again."
                )
            self.user_query = ""
            self.assistant_mode = False
        return True

    def render_gaussians(self, viewmat, scaling, anaglyph=False):
        outputs = super().render_gaussians(viewmat, scaling, anaglyph)
        if not self.assistant_mode:
            return outputs
        if isinstance(outputs, tuple):
            output_cv, left, right = outputs
            # Render the assistant text on the output image
            cv2.putText(
                output_cv,
                self.user_query,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            return (
                np.ascontiguousarray(output_cv),
                np.ascontiguousarray(left),
                np.ascontiguousarray(right),
            )
        else:
            output_cv = outputs
            # Render the assistant text on the output image
            cv2.putText(
                output_cv,
                self.user_query,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            return np.ascontiguousarray(output_cv)
        
        

def main(args: Args):
    format = args.format or args.rasterizer
    if args.rasterizer:
        warnings.warn(
            "`rasterizer` is deprecated. Use `format` instead.", DeprecationWarning
        )
    if not format:
        raise ValueError("Must specify --format or the deprecated --rasterizer")

    splats = load_checkpoint(args.checkpoint, args.data_dir, format, args.data_factor)
    splats = prune_by_gradients(splats)
    if args.lseg_checkpoint:
        splats["lseg"] = torch.load(args.lseg_checkpoint, map_location=device)
        print("splats['lseg'].shape:", splats["lseg"].shape)

    if args.turntable:
        viewer_args = ViewerArgs(turntable=True)
    else:
        viewer_args = ViewerArgs(turntable=False)

    viewer = ViewerWithAssistant(splats, viewer_args=viewer_args)
    viewer.run()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
