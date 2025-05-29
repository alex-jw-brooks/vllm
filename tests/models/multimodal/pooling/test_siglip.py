# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoProcessor

from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE

from ....conftest import ImageTestAssets
from ...utils import check_embeddings_close

# we use snapshot_download to prevent conflicts between
# dynamic_module and trust_remote_code for hf_runner
DOWNLOAD_PATTERN = ["*.json", "*.py", "*.safetensors", "*.txt", "*.model"]


@torch.inference_mode()
def run_siglip_test(
    image_assets: ImageTestAssets,
    model_id: str,
    vllm_runner,
    hf_runner,
    *,
    dtype: str,
):
    model = snapshot_download(model_id, allow_patterns=DOWNLOAD_PATTERN)
    torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[dtype]
    images = [asset.pil_image for asset in image_assets]

    with vllm_runner(model,
                     task="embed",
                     max_model_len=64,
                     dtype=dtype,
                     enforce_eager=False) as vllm_model:
        # HACK - currently we need to pass the prompt here, but it's unused.
        vllm_outputs = vllm_model.encode(["foo", "bar"], images=images)

    img_processor = AutoProcessor.from_pretrained(model)
    inputs = img_processor(images=images,
                           return_tensors='pt').to("cuda", dtype=torch_dtype)

    with hf_runner(model, dtype=torch_dtype, auto_cls=AutoModel) as hf_model:
        # Get the vision features; currently the vLLM siglip
        # implementation only contains the visual component for siglip.
        features = hf_model.model.get_image_features(**inputs)
        hf_outputs = [feature.detach().cpu() for feature in features]

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


# NOTE - siglip / siglip2 share the same implementation except for
# na-flex variants of siglip2; currently na-flex variants are unsupported.
@pytest.mark.parametrize("model_id", [
    "google/siglip-base-patch16-224",
    "google/siglip2-base-patch16-224",
])
@pytest.mark.parametrize("dtype", ["half"])
def test_models(dist_init, image_assets, model_id, dtype: str, vllm_runner,
                hf_runner) -> None:
    run_siglip_test(
        image_assets,
        model_id,
        vllm_runner,
        hf_runner,
        dtype=dtype,
    )
