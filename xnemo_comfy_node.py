from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import mediapipe as mp
import numpy as np
import torch
from diffusers import DDIMScheduler, AutoencoderKLTemporalDecoder
from omegaconf import OmegaConf
from PIL import Image
from scipy.ndimage import gaussian_filter1d

from .scripts.vid2pose import extract_bbox_mp
from src.models.motion_encoder.encoder import (
    MotEncoder_withExtra as MotEncoder,
)
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_motenc_long import Pose2VideoPipeline

try:
    from transformers import CLIPVisionModelWithProjection
except ImportError as exc:  # pragma: no cover - handled by requirements
    raise ImportError(
        "The transformers package is required to use the X-NeMo ComfyUI node."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = PROJECT_ROOT / "configs" / "prompts"
BLAZE_FACE_PATH = PROJECT_ROOT / "blaze_face_short_range.tflite"
PIPELINE_CACHE: Dict[Tuple[str, str], Tuple[Pose2VideoPipeline, torch.dtype, OmegaConf]] = {}


def _find_comfyui_root(start: Path) -> Path:
    current = start
    while current != current.parent:
        if (current / "models").exists() and (current / "custom_nodes").exists():
            return current
        current = current.parent
    return start


def _resolve_model_path(path_str: str, model_root: Path) -> Path:
    path = Path(path_str)
    parts = list(path.parts)
    if parts and parts[0] in {"pretrained_weights", "models"}:
        parts = parts[1:]
    return model_root.joinpath(*parts) if parts else model_root


def _load_config(config_name: str, comfy_root: Path) -> OmegaConf:
    config_path = CONFIG_ROOT / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config '{config_name}' was not found at {config_path}")

    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    comfy_models = comfy_root / "models"
    model_root = comfy_models / "x-nemo"
    if not model_root.exists():
        raise FileNotFoundError(
            "Expected X-NeMo weights inside 'models/x-nemo' under the ComfyUI root."
        )

    cfg.pretrained_base_model_path = str(
        _resolve_model_path(cfg.pretrained_base_model_path, model_root)
    )
    cfg.image_encoder_path = str(
        _resolve_model_path(cfg.image_encoder_path, model_root)
    )
    cfg.vae_path = str(_resolve_model_path(cfg.vae_path, model_root))
    cfg.denoising_unet_path = str(
        _resolve_model_path(cfg.denoising_unet_path, model_root)
    )
    cfg.temporal_module_path = str(
        _resolve_model_path(cfg.temporal_module_path, model_root)
    )

    inference_path = Path(cfg.inference_config)
    if not inference_path.is_absolute():
        inference_path = PROJECT_ROOT / inference_path
    cfg.inference_config = str(inference_path)

    return cfg


def _init_pipeline(
    config_name: str, device: torch.device
) -> Tuple[Pose2VideoPipeline, torch.dtype, OmegaConf]:
    cache_key = (config_name, str(device))
    if cache_key in PIPELINE_CACHE:
        return PIPELINE_CACHE[cache_key]

    comfy_root = _find_comfyui_root(PROJECT_ROOT)
    config = _load_config(config_name, comfy_root)

    if getattr(config, "weight_dtype", "fp16") == "fp16" and device.type == "cuda":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    vae = AutoencoderKLTemporalDecoder.from_pretrained(config.vae_path).to(
        device, dtype=weight_dtype
    )

    infer_config = OmegaConf.load(config.inference_config)
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path, subfolder="unet"
    ).to(device=device, dtype=weight_dtype)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device=device)

    motion_encoder = MotEncoder().to(dtype=weight_dtype, device=device)
    motion_encoder.eval()

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device=device)

    sched_kwargs = OmegaConf.to_container(
        infer_config.noise_scheduler_kwargs,
        resolve=True,
    )
    scheduler = DDIMScheduler(**sched_kwargs)

    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu", weights_only=True), strict=False
    )
    reference_weights_path = Path(config.denoising_unet_path).with_name(
        Path(config.denoising_unet_path).name.replace("denoising_unet", "reference_unet")
    )
    reference_unet.load_state_dict(
        torch.load(reference_weights_path, map_location="cpu", weights_only=True), strict=True
    )
    motion_encoder.load_state_dict(
        torch.load(
            Path(config.denoising_unet_path)
            .with_name(Path(config.denoising_unet_path).name.replace("denoising_unet", "motion_encoder")),
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    denoising_unet.load_state_dict(
        torch.load(config.temporal_module_path, map_location="cpu", weights_only=True), strict=False
    )

    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        motion_encoder=motion_encoder,
        scheduler=scheduler,
    )
    pipe = pipe.to(device, dtype=weight_dtype)
    PIPELINE_CACHE[cache_key] = (pipe, weight_dtype, config)
    return PIPELINE_CACHE[cache_key]


def _check_oob_new(bbox: Iterable[float], frame_shape: Tuple[int, int, int]):
    left, top, right, bot = bbox
    left_oob = -left
    right_oob = right - frame_shape[1]
    top_oob = -top
    bot_oob = bot - frame_shape[0]
    return (
        left_oob,
        right_oob,
        top_oob,
        bot_oob,
        max(left_oob, right_oob, top_oob, bot_oob) > 0,
    )


def _compute_bbox(
    tube_bbox: Tuple[float, float, float, float],
    frame_shape: Tuple[int, int, int],
    aspect_preserving: bool,
    increase_area: float,
) -> Tuple[bool, int, int, int, int]:
    def compute_aspect_preserved_bbox(bbox, delta):
        left, top, right, bot = bbox
        width = right - left
        height = bot - top
        width_increase = max(delta, ((1 + 2 * delta) * height - width) / (2 * width))
        height_increase = max(delta, ((1 + 2 * delta) * width - height) / (2 * height))
        left = int(left - width_increase * width)
        top = int(top - height_increase * height)
        right = int(right + width_increase * width)
        bot = int(bot + height_increase * height)
        return left, top, right, bot

    def compute_increased_bbox(bbox, delta):
        left, top, right, bot = bbox
        width = right - left
        height = bot - top
        left = int(left - delta * width)
        top = int(top - delta * height)
        right = int(right + delta * width)
        bot = int(bot + delta * height)
        return left, top, right, bot

    if aspect_preserving:
        left, top, right, bot = compute_aspect_preserved_bbox(tube_bbox, increase_area)
    else:
        left, top, right, bot = compute_increased_bbox(tube_bbox, increase_area)

    left_oob, right_oob, top_oob, bot_oob, oob_flag = _check_oob_new(
        (left, top, right, bot), frame_shape
    )
    if oob_flag:
        if left_oob > 0 and right_oob + left_oob <= 0:
            new_box = (left + left_oob, top, right + left_oob, bot)
            if not _check_oob_new(new_box, frame_shape)[-1]:
                left, top, right, bot = new_box
        elif right_oob > 0 and right_oob + left_oob <= 0:
            new_box = (left - right_oob, top, right - right_oob, bot)
            if not _check_oob_new(new_box, frame_shape)[-1]:
                left, top, right, bot = new_box
        if top_oob > 0 and top_oob + bot_oob <= 0:
            new_box = (left, top + top_oob, right, bot + top_oob)
            if not _check_oob_new(new_box, frame_shape)[-1]:
                left, top, right, bot = new_box
        elif bot_oob > 0 and top_oob + bot_oob <= 0:
            new_box = (left, top - bot_oob, right, bot - bot_oob)
            if not _check_oob_new(new_box, frame_shape)[-1]:
                left, top, right, bot = new_box
    return oob_flag, int(left), int(top), int(right), int(bot)


def _crop_bbox_from_frames(
    frame_list: List[np.ndarray],
    tube_bbox: Tuple[float, float, float, float],
    increase_area: float = 0.1,
    aspect_preserving: bool = True,
) -> Tuple[bool, List[np.ndarray], List[int]]:
    frame_shape = frame_list[0].shape
    oob_flag, left, top, right, bot = _compute_bbox(
        tube_bbox, frame_shape, aspect_preserving, increase_area
    )
    selected = [frame[top:bot, left:right] for frame in frame_list]
    return oob_flag, selected, [left, top, right, bot]


def _get_bbox_from_center(
    center: Tuple[float, float], length: Tuple[int, int], size: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    if length is None:
        raise RuntimeError("Failed to determine crop size from driving video.")
    center_x, center_y = center
    w, h = length
    left = center_x - w / 2
    top = center_y - h / 2
    right = center_x + w / 2
    bot = center_y + h / 2
    left_oob, right_oob, top_oob, bot_oob, oob_flag = _check_oob_new(
        (left, top, right, bot), (size[0], size[1], 3)
    )
    if oob_flag:
        x_offset = max(-left, 0) + min(size[1] - 1 - right, 0)
        y_offset = max(-top, 0) + min(size[0] - 1 - bot, 0)
        left += x_offset
        right += x_offset
        top += y_offset
        bot += y_offset
    return int(left), int(top), int(right), int(bot)


def _get_bbox_param(bbox: np.ndarray, ref_bbox: np.ndarray) -> np.ndarray:
    left, top, right, bot = bbox
    center = np.array([(bot + top) * 0.5, (left + right) * 0.5])
    length = max(right - left, bot - top)

    ref_left, ref_top, ref_right, ref_bot = ref_bbox
    ref_center = np.array([(ref_bot + ref_top) * 0.5, (ref_left + ref_right) * 0.5])
    ref_length = max(ref_right - ref_left, ref_bot - ref_top)

    return np.asarray(((center - ref_center) / ref_length).tolist() + [length / ref_length])


class XNemoPoseToVideo:
    """ComfyUI node that wraps the official X-NeMo pose-to-video pipeline."""

    @classmethod
    def INPUT_TYPES(cls):
        config_names = [p.stem for p in CONFIG_ROOT.glob("*.yaml")]
        config_names.sort()
        default_config = config_names[0] if config_names else "animation"
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "driving_video": ("IMAGE",),
                "config_name": (config_names or [default_config], {"default": default_config}),
                "width": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
                "height": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
                "steps": ("INT", {"default": 35, "min": 10, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "context_frames": ("INT", {"default": 24, "min": 8, "max": 48}),
                "context_overlap": ("INT", {"default": 4, "min": 0, "max": 24}),
                "fps": ("INT", {"default": 25, "min": 1, "max": 60}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {
                "smoothing_sigma": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "bbox_scale": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate"
    CATEGORY = "X-NeMo"

    def generate(
        self,
        reference_image: torch.Tensor,
        driving_video: torch.Tensor,
        config_name: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        context_frames: int,
        context_overlap: int,
        fps: int,
        seed: int,
        smoothing_sigma: float = 5.0,
        bbox_scale: float = 1.1,
    ):
        if reference_image.ndim != 4 or reference_image.shape[0] < 1:
            raise ValueError("Reference image tensor must have shape [N, H, W, C] with N >= 1.")
        if driving_video.ndim != 4 or driving_video.shape[0] < 1:
            raise ValueError("Driving video must be a 4D tensor [F, H, W, C] with at least one frame.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline, weight_dtype, _ = _init_pipeline(config_name, device)

        ref_img = (
            reference_image[0].detach().cpu().clamp(0, 1).mul(255).to(torch.uint8).numpy()
        )
        driving_np = (
            driving_video.detach().cpu().clamp(0, 1).mul(255).to(torch.uint8).numpy()
        )

        if not BLAZE_FACE_PATH.exists():
            raise FileNotFoundError(
                f"Expected BlazeFace model at {BLAZE_FACE_PATH}. Please keep the file with the node."
            )

        base_options = mp.tasks.BaseOptions(model_asset_path=str(BLAZE_FACE_PATH))
        detector_options = mp.tasks.vision.FaceDetectorOptions(base_options=base_options)
        img_detector = mp.tasks.vision.FaceDetector.create_from_options(detector_options)

        ref_bbox = extract_bbox_mp(ref_img, None, img_detector)
        if isinstance(ref_bbox, str):
            raise RuntimeError(
                f"Failed to extract face bounding box from reference image: {ref_bbox}"
            )

        ref_bbox = np.array(ref_bbox, dtype=np.float32)
        _, ref_crops, _ = _crop_bbox_from_frames(
            [ref_img], tuple(ref_bbox.tolist()), increase_area=0.5, aspect_preserving=True
        )
        ref_pose_image = ref_crops[0]
        ref_pose_pil = Image.fromarray(ref_pose_image).convert("RGB")
        ref_image_pil = Image.fromarray(ref_img).convert("RGB")

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        init_latents = None

        video_detector_options = mp.tasks.vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
        )
        detector = mp.tasks.vision.FaceDetector.create_from_options(video_detector_options)

        bbox_params: List[torch.Tensor] = []
        ani_bbox_center: List[List[float]] = []
        fix_length: Optional[Tuple[int, int]] = None
        timestamp_ms = 0

        for idx in range(driving_np.shape[0]):
            pose_image = driving_np[idx]
            timestamp_ms += int(1000 / max(fps, 1))
            bbox = extract_bbox_mp(pose_image, None, detector, timestamp_ms)
            if isinstance(bbox, str):
                raise RuntimeError(
                    f"Failed to extract face bounding box from frame {idx}: {bbox}"
                )

            bbox_np = np.array(bbox, dtype=np.float32)
            bbox_param = torch.from_numpy(_get_bbox_param(bbox_np, ref_bbox))
            bbox_param = torch.ones_like(bbox_param)
            bbox_param[:2] *= 0
            bbox_params.append(bbox_param.to(device=device, dtype=weight_dtype))

            if idx == 0:
                width_len = round((bbox_np[2] - bbox_np[0]) * bbox_scale) // 2 * 2
                height_len = round((bbox_np[3] - bbox_np[1]) * bbox_scale) // 2 * 2
                fix_length = (max(width_len, 2), max(height_len, 2))
            left, top, right, bot = bbox_np
            center_x = (left + right) * 0.5
            center_y = (top + bot) * 0.5
            ani_bbox_center.append([center_x, center_y])

        ani_bbox_center = np.asarray(ani_bbox_center, dtype=np.float32)
        if smoothing_sigma > 0:
            ani_bbox_center = gaussian_filter1d(
                ani_bbox_center, sigma=smoothing_sigma, axis=0
            )
        bbox_params = torch.stack(bbox_params, dim=0)

        pose_images: List[Image.Image] = []
        for idx in range(driving_np.shape[0]):
            pose_image = driving_np[idx]
            bbox = _get_bbox_from_center(
                tuple(ani_bbox_center[idx]), fix_length, pose_image.shape[:2]
            )
            left, top, right, bot = bbox
            pose_crop = pose_image[int(top) : int(bot), int(left) : int(right)]
            pose_pil = Image.fromarray(pose_crop).convert("RGB")
            pose_pil = pose_pil.resize((width, height))
            pose_images.append(pose_pil)

        pipeline.set_progress_bar_config(disable=True)
        output = pipeline(
            ref_image_pil,
            pose_images,
            ref_pose_pil,
            width,
            height,
            len(pose_images),
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            init_latents=init_latents,
            mot_bbox_param=bbox_params,
            context_frames=context_frames,
            context_overlap=context_overlap,
        ).videos

        if isinstance(output, np.ndarray):
            video = torch.from_numpy(output)
        else:
            video = output

        if video.ndim != 5:
            raise RuntimeError(
                f"Unexpected video tensor shape from pipeline: {tuple(video.shape)}"
            )
        video = video.to(dtype=torch.float32)
        video = video[0].permute(1, 2, 3, 0).contiguous()
        video = video.clamp(0.0, 1.0)
        return (video,)
