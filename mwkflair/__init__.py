"""
MWK-FLAIR: FLAIR with HarMA Adapter Integration
A parameter-efficient fine-tuning framework for vision-language models.
"""

from .model import FLAIR, CLIPVisionCfg, CLIPTextCfg
from .harma_modules import (
    BiShareAdapter,
    MultimodalGatedAdapter,
    AdaptiveTripletLoss,
    StandardTripletLoss,
    build_harma_adapters
)
from .harma_adapter import (
    HarmaAdapter,
    MultiHeadHarmaAdapter,
    HarmaAdapterLayer,
    freeze_model_except_adapters
)
from .factory import (
    create_model,
    create_model_and_transforms,
    get_tokenizer,
    get_model_config,
    load_checkpoint
)

__version__ = "1.0.0"
__author__ = "MWK"

__all__ = [
    # Core Model
    "FLAIR",
    "CLIPVisionCfg",
    "CLIPTextCfg",
    # HarMA Modules
    "BiShareAdapter",
    "MultimodalGatedAdapter",
    "AdaptiveTripletLoss",
    "StandardTripletLoss",
    "build_harma_adapters",
    # HarMA Adapter
    "HarmaAdapter",
    "MultiHeadHarmaAdapter",
    "HarmaAdapterLayer",
    "freeze_model_except_adapters",
    # Factory
    "create_model",
    "create_model_and_transforms",
    "get_tokenizer",
    "get_model_config",
    "load_checkpoint",
]
