from .xnemo_comfy_node import XNemoLoadModels, XNemoPoseToVideo

NODE_CLASS_MAPPINGS = {
    "XNemoLoadModels": XNemoLoadModels,
    "XNemoPoseToVideo": XNemoPoseToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XNemoLoadModels": "X-NeMo Load Models",
    "XNemoPoseToVideo": "X-NeMo Pose-to-Video",
}
