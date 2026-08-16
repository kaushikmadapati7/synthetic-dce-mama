"""Per-model training loops, dispatched by the TRAINERS registry."""
from .gan import train_gan
from .ldm_ddpm import train_ldm_ddpm
from .ldm_flow import train_ldm_flow
from .unet import train_unet
from .refine import train_refine
from .teacher import train_teacher
from .multitask import train_multitask
from .segmenter import train_segmenter
from ._ldm_base import train_ldm_bridge

TRAINERS = {
    "gan": train_gan,
    "ldm_ddpm": train_ldm_ddpm,
    "ldm_flow": train_ldm_flow,
    "ldm_bridge": train_ldm_bridge,
    "unet": train_unet,
    "refine": train_refine,
    "teacher": train_teacher,
    "multitask": train_multitask,
    "segmenter": train_segmenter,
}

__all__ = ["TRAINERS", "train_gan", "train_ldm_ddpm", "train_ldm_flow",
           "train_ldm_bridge", "train_unet", "train_refine", "train_teacher",
           "train_multitask", "train_segmenter"]
