# SPDX-License-Identifier: Apache-2.0
"""Implementation of SiglipVisionModel intended to be only used
within a vision language model."""

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Optional, Union

import torch
from torch import nn
from transformers import BatchFeature, SiglipVisionConfig

from vllm.attention.layer import MultiHeadAttention
from vllm.config import VllmConfig
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptIndexTargets,
                                        PromptInsertion, PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import (IntermediateTensors, PoolerOutput,
                           PoolingSequenceGroupOutput)

from .interfaces import SupportsMultiModal, SupportsV0Only
from .vision import VisionEncoderInfo, resolve_visual_encoder_outputs

logger = init_logger(__name__)


class SiglipProcessingInfo(BaseProcessingInfo):

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}


class SiglipEncoderInfo(VisionEncoderInfo[SiglipVisionConfig]):

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        return self.get_patch_grid_length()**2

    def get_image_size(self) -> int:
        return self.vision_config.image_size

    def get_patch_size(self) -> int:
        return self.vision_config.patch_size

    def get_patch_grid_length(self) -> int:
        image_size, patch_size = self.get_image_size(), self.get_patch_size()
        return image_size // patch_size


# Adapted from https://github.com/huggingface/transformers/blob/v4.43.3/src/transformers/models/siglip/modeling_siglip.py#L249 # noqa
class SiglipVisionEmbeddings(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size)**2
        self.num_positions = self.num_patches
        self.position_embedding = VocabParallelEmbedding(
            self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions, dtype=torch.int64).expand(
                (1, -1)),
            persistent=False,
        )

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int,
                                 width: int) -> torch.Tensor:
        """
        This method is an adapted method for SigLIP (due to SigLIP not having
        class embedding unlike other ViTs) that allows the model to interpolate
        the pre-trained position encodings such that it can be usable on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """
        position_embeddings = self.position_embedding.weight.unsqueeze(0)
        num_patches = embeddings.shape[1]
        num_positions = position_embeddings.shape[1]
        if num_patches == num_positions and height == width:
            return position_embeddings

        dim = embeddings.shape[-1]
        height = height // self.patch_size
        width = width // self.patch_size
        # we add a small number to avoid floating point error
        # in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        height, width = height + 0.1, width + 0.1

        patch_pos_embed = position_embeddings.reshape(
            1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)),
            dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(
                height / math.sqrt(num_positions),
                width / math.sqrt(num_positions),
            ),
            mode="bicubic",
            align_corners=False,
        )
        if (int(height) != patch_pos_embed.shape[-2]
                or int(width) != patch_pos_embed.shape[-1]):
            raise ValueError("Width or height does not match with "
                             "the interpolated position embeddings")

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(self,
                pixel_values: torch.Tensor,
                interpolate_pos_encoding: bool = False) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(
            dtype=target_dtype))  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(
                embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(
                self.position_ids)
        return embeddings


class SiglipAttention(nn.Module):

    def __init__(
        self,
        config: SiglipVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got "
                             "`embed_dim`: {self.embed_dim} and `num_heads`:"
                             f" {self.num_heads}).")

        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.out_proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_per_partition = divide(self.num_heads, self.tp_size)

        self.attn = MultiHeadAttention(self.num_heads_per_partition,
                                       self.head_dim, self.scale)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""
        qkv_states, _ = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv_states.chunk(3, dim=-1)

        out = self.attn(query_states, key_states, value_states)
        attn_output, _ = self.out_proj(out)

        return attn_output, None


class SiglipMLP(nn.Module):

    def __init__(
        self,
        config: SiglipVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        # Special handling for BNB and torchao quantization
        if quant_config and quant_config.get_name() in [
                "bitsandbytes", "torchao"
        ]:
            quantizable = True
        else:
            # For other quantization, we require the hidden size to be a
            # multiple of 64
            quantizable = (config.hidden_size % 64 == 0
                           and config.intermediate_size % 64 == 0)
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config if quantizable else None,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            quant_config=quant_config if quantizable else None,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):

    def __init__(
        self,
        config: SiglipVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.embed_dim = config.hidden_size

        self.self_attn = SiglipAttention(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim,
                                        eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim,
                                        eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, None


class SiglipEncoder(nn.Module):

    def __init__(
        self,
        config: SiglipVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        num_hidden_layers_override: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override

        self.layers = nn.ModuleList([
            SiglipEncoderLayer(config,
                               quant_config=quant_config,
                               prefix=f"{prefix}.layers.{layer_idx}")
            for layer_idx in range(num_hidden_layers)
        ])

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        return_all_hidden_states: bool,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        hidden_states_pool = [inputs_embeds]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            hidden_states, _ = encoder_layer(hidden_states)
            if return_all_hidden_states:
                hidden_states_pool.append(hidden_states)
        # If we have multiple feature sample layers, we return all hidden
        # states in order and grab the ones we need by index.
        if return_all_hidden_states:
            return hidden_states_pool
        return hidden_states


class SiglipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(
        self,
        config: SiglipVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        # TODO(ChristopherCho): Implement vLLM version of MultiheadAttention
        self.attention = torch.nn.MultiheadAttention(
            config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config=config,
                             quant_config=quant_config,
                             prefix=f"{prefix}.mlp")

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class SiglipVisionTransformer(nn.Module):

    def __init__(
        self,
        config: SiglipVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        require_post_norm: Optional[bool] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)

        self.encoder = SiglipEncoder(
            config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            prefix=f"{prefix}.encoder",
        )

        num_hidden_layers = config.num_hidden_layers
        if len(self.encoder.layers) > config.num_hidden_layers:
            raise ValueError(
                f"The original encoder only has {num_hidden_layers} "
                f"layers, but you requested {len(self.encoder.layers)} layers."
            )

        # If possible, skip post_layernorm to conserve memory
        if require_post_norm is None:
            require_post_norm = len(self.encoder.layers) == num_hidden_layers

        if require_post_norm:
            self.post_layernorm = nn.LayerNorm(embed_dim,
                                               eps=config.layer_norm_eps)
        else:
            self.post_layernorm = None

        self.use_head = (True if not hasattr(config, "vision_use_head") else
                         config.vision_use_head)
        if self.use_head:
            self.head = SiglipMultiheadAttentionPoolingHead(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.head",
            )

    def forward(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = True,
        feature_sample_layers: Optional[list[int]] = None,
    ) -> torch.Tensor:

        hidden_states = self.embeddings(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        return_all_hidden_states = feature_sample_layers is not None

        # Produces either the last layer output or all of the hidden states,
        # depending on if we have feature_sample_layers or not
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            return_all_hidden_states=return_all_hidden_states,
        )

        # Handle post-norm (if applicable) and stacks feature layers if needed
        encoder_outputs = resolve_visual_encoder_outputs(
            encoder_outputs, feature_sample_layers, self.post_layernorm,
            self.config.num_hidden_layers)

        # Apply multi-headed attention pooling if we have a head
        if self.use_head:
            encoder_outputs = self.head(encoder_outputs)

        return encoder_outputs


class SiglipInputBuilder(BaseDummyInputsBuilder[SiglipProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        # NOTE - currently this is unused since running as a
        # standalone embedding model is only on v0.
        num_images = mm_counts.get("image", 0)

        return {
            "image":
            self._get_dummy_images(
                width=224,
                height=224,
                num_images=num_images,
            )
        }


class SiglipMultiModalProcessor(BaseMultiModalProcessor):
    """Siglip's multimodal processor; note that currently, we only implement
    vision encoding, so the placeholders for the prompt are unused.

    TODO: Revisit how to support Siglip's text encoder with respect to the
    encode() API
    """

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(pixel_values=MultiModalFieldConfig.batched("image"), )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        # HACK - currently we need to insert something for the prompt update
        # to be happy, but we don't actually use the text prompt in this
        # implementation.
        return [
            PromptInsertion(
                modality="image",
                target=PromptIndexTargets.start(),
                insertion=[1],
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(SiglipMultiModalProcessor,
                                        info=SiglipProcessingInfo,
                                        dummy_inputs=SiglipInputBuilder)
class SiglipVisionModel(nn.Module, SupportsMultiModal, SupportsV0Only):
    config_class = SiglipVisionConfig
    main_input_name = "pixel_values"

    def __init__(
        self,
        vllm_config: Optional[VllmConfig] = None,
        config: Optional[SiglipVisionConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        require_post_norm: Optional[bool] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # siglip may be initialized directly, or used as a vision tower in VLMs
        if vllm_config is not None:
            if config is not None:
                logger.warning(
                    "config & vllm_config both provided to initialize siglip; "
                    "using config")
            else:
                hf_config = vllm_config.model_config.hf_config
                config = hf_config.vision_config

            # Only have a pooler if we are running the model directly.
            # Note that this is the vLLM pooler, which is distinct from
            # the Siglip's MHA pooling head.
            self._pooler = Pooler.from_config_with_defaults(
                vllm_config.model_config.pooler_config,
                pooling_type=PoolingType.ALL,
                normalize=False,
                softmax=False)
        else:
            # This model is being initialized within a VLM.
            self._pooler = None

        self.vision_model = SiglipVisionTransformer(
            config,
            quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            require_post_norm=require_post_norm,
            prefix=f"{prefix}.vision_model",
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        interpolate_pos_encoding: bool = False,
        feature_sample_layers: Optional[list[int]] = None,
    ) -> torch.Tensor:
        if pixel_values is None:
            raise ValueError("Siglip vision model requires pixel_values")
        if len(pixel_values.shape) == 5:
            # We may have an extra singleton dimension from batch stacking
            pixel_values = pixel_values.squeeze(axis=1)

        return self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            feature_sample_layers=feature_sample_layers,
        )

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        # Shouldn't happen since standalone instances
        # of this class should create the pooler.
        pooled_outputs = [
            PoolingSequenceGroupOutput(data) for data in hidden_states
        ]
        return PoolerOutput(outputs=pooled_outputs)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        layer_count = len(self.vision_model.encoder.layers)

        for name, loaded_weight in weights:
            # vLLM currently only implements the vision part of siglip
            if (name.startswith("text_model")
                    or name in ["logit_scale", "logit_bias"]):
                continue

            # post_layernorm is optional in SiglipVisionModel
            if (name.startswith("vision_model.post_layernorm")
                    and self.vision_model.post_layernorm is None):
                continue

            # omit layers when num_hidden_layers_override is set
            if name.startswith("vision_model.encoder.layers"):
                layer_idx = int(name.split(".")[3])
                if layer_idx >= layer_count:
                    continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
