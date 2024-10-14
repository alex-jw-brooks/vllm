"""Common tests for testing .generate() functionality for single / multiple
image support for different VLMs in vLLM.
"""
import os
from pathlib import PosixPath
from typing import Optional

import pytest
from transformers import AutoModelForVision2Seq, AutoTokenizer, BatchEncoding

from vllm.utils import identity, is_cpu, is_hip, cuda_device_count_stateless

from ....conftest import _ImageAssets, IMAGE_ASSETS
from ...utils import check_outputs_equal
from ....utils import get_memory_gb, fork_new_process_for_each_test
from . import utils as vlm_utils
from .vlm_test_types import (CustomTestOptions, ImageSizeWrapper,
                             VLMTestInfo, VlmTestType)

# This hack is needed for phi3v & paligemma models
# ROCm Triton FA can run into shared memory issues with these models,
# use other backends in the meantime
# FIXME (mattwong, gshtrasb, hongxiayan)
if is_hip():
    os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"


COMMON_BROADCAST_SETTINGS = {
    "test_type": VlmTestType.NEW_PROC_IMAGE,
    "dtype": "half",
    "max_tokens": 5,
    "tensor_parallel_size": 2,
    "image_size_factors": ((.25, 0.5, 1.0),),
    "distributed_executor_backend": ("ray", "mp"),
    "skip": cuda_device_count_stateless() < 2,
}

### Test configuration for specific models;
# NOTE: the key in the dict below is not mostly used as an identifier;
# it will be first in all of the expanded parametrizations, so it will
# tell you which test configuration failed.

# yapf: disable
VLM_TEST_SETTINGS = {
    "blip2": VLMTestInfo(
        models="Salesforce/blip2-opt-2.7b",
        prompt_formatter=lambda img_prompt: f"Question: {img_prompt} Answer:",
        test_type=VlmTestType.IMAGE,
        img_idx_to_prompt=lambda idx: "",
        auto_cls=AutoModelForVision2Seq,
        vllm_output_post_proc=vlm_utils.blip2_vllm_to_hf_output,
    ),
    "chameleon": VLMTestInfo(
        models="facebook/chameleon-7b",
        prompt_formatter=lambda img_prompt: f"USER: {img_prompt}\nASSISTANT:",
        test_type=VlmTestType.IMAGE,
        max_model_len=4096,
        auto_cls=AutoModelForVision2Seq,
        postprocess_inputs=vlm_utils.get_key_type_post_processor(
            "pixel_values",
            "bfloat16"
        ),
        # For chameleon, we only compare the sequences
        vllm_output_post_proc = lambda vllm_output, model: vllm_output[:2],
        hf_output_post_proc = lambda hf_output, model: hf_output[:2],
        comparator=check_outputs_equal,
        max_tokens=8,
        dtype="bfloat16",
    ),
    "fuyu": VLMTestInfo(
        models="adept/fuyu-8b",
        prompt_formatter=lambda img_prompt: f"{img_prompt}\n",
        test_type=VlmTestType.IMAGE,
        img_idx_to_prompt=lambda idx: "",
        max_model_len=2048,
        max_num_seqs=2,
        use_tokenizer_eos=True,
        vllm_output_post_proc=vlm_utils.fuyu_vllm_to_hf_output,
        num_logprobs=10,
        dtype="bfloat16" if is_cpu() else "half",
        image_size_factors=((), (0.25,), (0.25, 0.25, 0.25), (0.25, 0.2, 0.15)),
    ),
    "glm4": VLMTestInfo(
        models="THUDM/glm-4v-9b",
        prompt_formatter=identity,
        test_type=VlmTestType.NEW_PROC_IMAGE,
        img_idx_to_prompt=lambda idx: "",
        max_model_len=2048,
        max_num_seqs=2,
        dtype="bfloat16",
        get_stop_token_ids=lambda tok: [151329, 151336, 151338],
        skip=(get_memory_gb() < 48), # large GPU test; run in separate proc
        patch_hf_runner=vlm_utils.glm_patch_hf_runner,
    ),
    "llava": VLMTestInfo(
        models="llava-hf/llava-1.5-7b-hf",
        prompt_formatter=lambda img_prompt: f"USER: {img_prompt}\nASSISTANT:",
        test_type=(
            VlmTestType.EMBEDDING,
            VlmTestType.IMAGE,
            VlmTestType.CUSTOM_INPUTS
        ),
        convert_assets_to_embeddings=vlm_utils.get_llava_embeddings,
        max_model_len=4096,
        auto_cls=AutoModelForVision2Seq,
        vllm_output_post_proc=vlm_utils.llava_image_vllm_to_hf_output,
        custom_test_opts=CustomTestOptions(
            inputs=vlm_utils.multi_image_multi_aspect_ratio_inputs_llava(is_llava=True),
            limit_mm_per_prompt={"image": 4},
        ),
    ),
    "llava-next": VLMTestInfo(
        models="llava-hf/llava-v1.6-mistral-7b-hf",
        prompt_formatter=lambda img_prompt: f"[INST] {img_prompt} [/INST]",
        test_type=(VlmTestType.IMAGE, VlmTestType.CUSTOM_INPUTS),
        max_model_len=10240,
        auto_cls=AutoModelForVision2Seq,
        vllm_output_post_proc=vlm_utils.llava_image_vllm_to_hf_output,
        custom_test_opts=CustomTestOptions(
            inputs=vlm_utils.multi_image_multi_aspect_ratio_inputs_llava(is_llava=False),
            limit_mm_per_prompt={"image": 4},
        ),
        # Llava-next tests fixed sizes & the default size factors
        image_sizes=(((1669, 2560), (2560, 1669), (183, 488), (488, 183),),),
    ),
    "minicpmv": VLMTestInfo(
        models="openbmb/MiniCPM-Llama3-V-2_5",
        prompt_formatter=lambda img_prompt: f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{img_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",  # noqa: E501
        test_type=(VlmTestType.MULTI_IMAGE),
        img_idx_to_prompt=lambda idx: "(<image>./</image>)\n",
        max_model_len=4096,
        max_num_seqs=2,
        get_stop_token_ids=lambda tok: [tok.eos_id, tok.eot_id],
        postprocess_inputs=vlm_utils.wrap_inputs_post_processor,
        hf_output_post_proc=vlm_utils.minicmpv_trunc_hf_output,
    ),
    "paligemma": VLMTestInfo(
        models="google/paligemma-3b-mix-224",
        prompt_formatter=identity,
        test_type=VlmTestType.IMAGE,
        img_idx_to_prompt = lambda idx: "",
        # Paligemma uses its own sample prompts because the default one fails
        single_image_prompts=IMAGE_ASSETS.prompts({
            "stop_sign": "caption es",
            "cherry_blossom": "What is in the picture?",
        }),
        auto_cls=AutoModelForVision2Seq,
        vllm_output_post_proc=vlm_utils.paligemma_vllm_to_hf_output,
        dtype="half" if is_hip() else ("half", "float"),
    ),
    "phi3v": VLMTestInfo(
        models="microsoft/Phi-3.5-vision-instruct",
        prompt_formatter=lambda img_prompt: f"<|user|>\n{img_prompt}<|end|>\n<|assistant|>\n", # noqa: E501
        test_type=(VlmTestType.IMAGE, VlmTestType.MULTI_IMAGE),
        img_idx_to_prompt=lambda idx: f"<|image_{idx}|>\n",
        max_model_len=4096,
        max_num_seqs=2,
        # use eager mode for hf runner, since phi3v didn't work with flash_attn
        model_kwargs={"_attn_implementation": "eager"},
        use_tokenizer_eos=True,
        vllm_output_post_proc=vlm_utils.phi3v_vllm_to_hf_output,
        num_logprobs=10,
    ),
    "qwen": VLMTestInfo(
        models="Qwen/Qwen-VL",
        prompt_formatter=identity,
        test_type=(VlmTestType.IMAGE, VlmTestType.MULTI_IMAGE),
        img_idx_to_prompt=lambda idx: f"Picture {idx}: <img></img>\n",
        max_model_len=1024,
        max_num_seqs=2,
        vllm_output_post_proc=vlm_utils.qwen_vllm_to_hf_output,
        prompt_path_encoder=vlm_utils.qwen_prompt_path_encoder,
    ),
    # Tests above this point have been validated to align with current tests
    "intern_vl": VLMTestInfo(
        models=("OpenGVLab/InternVL2-1B", "OpenGVLab/InternVL2-2B"),
        prompt_formatter=lambda img_prompt: f"<|im_start|>User\n{img_prompt}<|im_end|>\n<|im_start|>Assistant\n", # noqa: E501
        test_type=(VlmTestType.IMAGE, VlmTestType.MULTI_IMAGE),
        max_model_len=4096,
        num_logprobs=10,
        dtype="bfloat16" if is_cpu() else "half",
        use_tokenizer_eos=True,
        patch_hf_runner=vlm_utils.internvl_patch_hf_runner,
    ),
    # Tensor parallel / multi-gpu broadcast tests
    "broadcast-chameleon": VLMTestInfo(
        models="facebook/chameleon-7b",
        prompt_formatter=lambda img_prompt: f"USER: {img_prompt}\nASSISTANT:",
        max_model_len=4096,
        auto_cls=AutoModelForVision2Seq,
        postprocess_inputs=vlm_utils.get_key_type_post_processor(
            "pixel_values", "half"
        ),
        vllm_output_post_proc = lambda vllm_output, model: vllm_output[:2],
        hf_output_post_proc = lambda hf_output, model: hf_output[:2],
        comparator=check_outputs_equal,
        **COMMON_BROADCAST_SETTINGS,
    ),
    "broadcast-llava": VLMTestInfo(
        models="llava-hf/llava-1.5-7b-hf",
        prompt_formatter=lambda img_prompt: f"USER: {img_prompt}\nASSISTANT:",
        max_model_len=4096,
        auto_cls=AutoModelForVision2Seq,
        vllm_output_post_proc=vlm_utils.llava_image_vllm_to_hf_output,
        **COMMON_BROADCAST_SETTINGS,
    ),
    "broadcast-llava-next": VLMTestInfo(
        models="llava-hf/llava-v1.6-mistral-7b-hf",
        prompt_formatter=lambda img_prompt: f"[INST] {img_prompt} [/INST]",
        max_model_len=10240,
        auto_cls=AutoModelForVision2Seq,
        vllm_output_post_proc=vlm_utils.llava_image_vllm_to_hf_output,
        **COMMON_BROADCAST_SETTINGS,
    )
}
# yapf: enable

### Test wrappers
# Wrappers around the core test running func for:
# - single image
# - multi-image
# - image embeddings
# - video [TODO]
# All wrappers (except single image) have a filter for dropping
# models that don't have applicable tests, and expanding the
# relevant VLMTestInfo object into a combination that can be
# consumed by parametrize()
@pytest.mark.parametrize(
    "model_type,model,max_tokens,num_logprobs,dtype,size_wrapper",
    vlm_utils.get_parametrized_options(VLM_TEST_SETTINGS,
                                       test_type=VlmTestType.IMAGE))
def test_single_image_generation(tmp_path: PosixPath, model_type: str,
                                 model: str, max_tokens: int,
                                 num_logprobs: int, dtype: str,
                                 size_wrapper: ImageSizeWrapper, 
                                 hf_runner,
                                 vllm_runner, image_assets: _ImageAssets):
    # Grab the model type's global model config to leverage callables
    test_info = VLM_TEST_SETTINGS[model_type]
    inputs = vlm_utils.build_single_image_inputs_from_test_info(
        test_info, image_assets, size_wrapper, tmp_path
    )

    vlm_utils.run_test(
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        inputs=inputs,
        model=model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        limit_mm_per_prompt={"image": 1},
        size_factors=size_wrapper,
        **test_info.get_non_parametrized_runner_kwargs()
    )


@pytest.mark.parametrize(
    "model_type,model,max_tokens,num_logprobs,dtype,distributed_executor_backend,size_wrapper",
    vlm_utils.get_parametrized_options(VLM_TEST_SETTINGS,
                                       test_type=VlmTestType.NEW_PROC_IMAGE))
@fork_new_process_for_each_test
def test_resource_heavy_image_generation(
                                 tmp_path: PosixPath, model_type: str,
                                 model: str, max_tokens: int,
                                 num_logprobs: int, dtype: str,
                                 distributed_executor_backend: Optional[str],
                                 size_wrapper: ImageSizeWrapper, 
                                 hf_runner,
                                 vllm_runner, image_assets: _ImageAssets):
    # Grab the model type's global model config to leverage callables
    test_info = VLM_TEST_SETTINGS[model_type]
    inputs = vlm_utils.build_single_image_inputs_from_test_info(
        test_info, image_assets, size_wrapper, tmp_path
    )

    vlm_utils.run_test(
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        inputs=inputs,
        model=model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        limit_mm_per_prompt={"image": 1},
        size_factors=size_wrapper,
        distributed_executor_backend=distributed_executor_backend,
        tensor_parallel_size=test_info.tensor_parallel_size,
        **test_info.get_non_parametrized_runner_kwargs()
    )


@pytest.mark.parametrize(
    "model_type,model,max_tokens,num_logprobs,dtype,size_wrapper",
    vlm_utils.get_parametrized_options(VLM_TEST_SETTINGS,
                                       test_type=VlmTestType.EMBEDDING))
def test_embedding_generation(model_type: str, model: str, max_tokens: int,
                              num_logprobs: int, dtype: str,
                              size_wrapper: ImageSizeWrapper, hf_runner,
                              vllm_runner, image_assets: _ImageAssets):
    # Grab the model type's global model config to leverage callables
    test_info = VLM_TEST_SETTINGS[model_type]
    inputs, vllm_embeddings = vlm_utils.build_embedding_inputs_from_test_info(
        test_info, image_assets, size_wrapper
    )

    vlm_utils.run_test(
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        inputs=inputs,
        model=model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        limit_mm_per_prompt={"image": 1},
        size_factors=size_wrapper,
        vllm_embeddings=vllm_embeddings,
        **test_info.get_non_parametrized_runner_kwargs()
    )


@pytest.mark.parametrize(
    "model_type,model,max_tokens,num_logprobs,dtype,size_wrapper",
    vlm_utils.get_parametrized_options(VLM_TEST_SETTINGS,
                                       test_type=VlmTestType.MULTI_IMAGE))
def test_multi_image_generation(tmp_path: PosixPath, model_type: str,
                                model: str, max_tokens: int, num_logprobs: int,
                                dtype: str,
                                size_wrapper: ImageSizeWrapper,
                                hf_runner, vllm_runner,
                                image_assets: _ImageAssets):
    # Grab the model type's global model config to leverage callables
    test_info = VLM_TEST_SETTINGS[model_type]
    inputs = vlm_utils.build_multi_image_inputs_from_test_info(
        test_info, image_assets, size_wrapper, tmp_path
    )

    vlm_utils.run_test(
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        inputs=inputs,
        model=model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        limit_mm_per_prompt={"image": len(image_assets)},
        size_factors=size_wrapper,
        **test_info.get_non_parametrized_runner_kwargs()
    )


@pytest.mark.parametrize("model_type,model,max_tokens,num_logprobs,dtype",
                         vlm_utils.get_parametrized_options(
                             VLM_TEST_SETTINGS,
                             test_type=VlmTestType.CUSTOM_INPUTS))
def test_custom_inputs(model_type: str, model: str, max_tokens: int,
                       num_logprobs: int, dtype: str,
                       hf_runner, vllm_runner):
    test_info = VLM_TEST_SETTINGS[model_type]
    custom_test_opts = test_info.custom_test_opts
    # Custom test cases can provide inputs directly, but they need to
    # explicitly provided a CustomTestConfig, which wraps the inputs and
    # the limit_mm_per_prompt
    assert custom_test_opts is not None

    inputs = custom_test_opts.inputs
    limit_mm_per_prompt = custom_test_opts.limit_mm_per_prompt

    assert inputs is not None and limit_mm_per_prompt is not None
    vlm_utils.run_test(
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        inputs=inputs,
        model=model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        limit_mm_per_prompt=limit_mm_per_prompt,
        size_factors=None,
        **test_info.get_non_parametrized_runner_kwargs()
    )
