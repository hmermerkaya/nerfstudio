# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Put all the method implementations in one location.
"""

from __future__ import annotations

from typing import Dict

import tyro
from nerfacc import ContractionType

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config_render_view import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.datamanagers.depth_datamanager import DepthDataManagerConfig
from nerfstudio.data.datamanagers.semantic_datamanager import SemanticDataManagerConfig
from nerfstudio.data.datamanagers.variable_res_datamanager import (
    VariableResDataManagerConfig,
)
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.data.dataparsers.dycheck_dataparser import DycheckDataParserConfig
from nerfstudio.data.dataparsers.friends_dataparser import FriendsDataParserConfig
from nerfstudio.data.dataparsers.instant_ngp_dataparser import (
    InstantNGPDataParserConfig,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.phototourism_dataparser import (
    PhototourismDataParserConfig,
)
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.render_viewer import TrainerConfig
from nerfstudio.engine.schedulers import SchedulerConfig
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.models.depth_nerfacto import DepthNerfactoModelConfig
from nerfstudio.models.instant_ngp import InstantNGPModelConfig
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.models.nerfacto_render_view import NerfactoModelConfig
from nerfstudio.models.nerfplayer_nerfacto import NerfplayerNerfactoModelConfig
from nerfstudio.models.nerfplayer_ngp import NerfplayerNGPModelConfig
from nerfstudio.models.semantic_nerfw import SemanticNerfWModelConfig
from nerfstudio.models.tensorf import TensoRFModelConfig
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.pipelines.base_pipeline_render_view import VanillaPipelineConfig
from nerfstudio.pipelines.dynamic_batch_render_view import DynamicBatchPipelineConfig

method_configs: Dict[str, TrainerConfig] = {}
descriptions = {
    "nerfacto": "Recommended real-time model tuned for real captures. This model will be continually updated.",
    "depth-nerfacto": "Nerfacto with depth supervision.",
    "instant-ngp": "Implementation of Instant-NGP. Recommended real-time model for unbounded scenes.",
    "instant-ngp-bounded": "Implementation of Instant-NGP. Recommended for bounded real and synthetic scenes",
    "mipnerf": "High quality model for bounded scenes. (slow)",
    "semantic-nerfw": "Predicts semantic segmentations and filters out transient objects.",
    "vanilla-nerf": "Original NeRF model. (slow)",
    "tensorf": "tensorf",
    "dnerf": "Dynamic-NeRF model. (slow)",
    "phototourism": "Uses the Phototourism data.",
    "nerfplayer-nerfacto": "NeRFPlayer with nerfacto backbone.",
    "nerfplayer-ngp": "NeRFPlayer with InstantNGP backbone.",
}

method_configs["nerfacto-render"] = TrainerConfig(
    method_name="nerfacto-render",
    pipeline=VanillaPipelineConfig(
        # datamanager=VanillaDataManagerConfig(
        #     dataparser=NerfstudioDataParserConfig(),
        #     train_num_rays_per_batch=4096,
        #     eval_num_rays_per_batch=4096,
        #     camera_optimizer=CameraOptimizerConfig(
        #         mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
        #     ),
        # ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["nerfacto"] = TrainerConfig(
    method_name="nerfacto",
    pipeline=VanillaPipelineConfig(
        # datamanager=VanillaDataManagerConfig(
        #     dataparser=NerfstudioDataParserConfig(),
        #     train_num_rays_per_batch=4096,
        #     eval_num_rays_per_batch=4096,
        #     camera_optimizer=CameraOptimizerConfig(
        #         mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
        #     ),
        # ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=method_configs, descriptions=descriptions)
    ]
]
"""Union[] type over config types, annotated with default instances for use with
tyro.cli(). Allows the user to pick between one of several base configurations, and
then override values in it."""
