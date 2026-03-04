# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test async beam search with audio models (encoder-decoder and decoder-only)."""

import io
import json

import openai
import pytest
import pytest_asyncio

from vllm.assets.audio import AudioAsset

from ...utils import RemoteOpenAIServer

WHISPER_MODEL_NAME = "openai/whisper-large-v3-turbo"


@pytest.fixture
def mary_had_lamb():
    """Audio fixture for Mary Had a Little Lamb."""
    path = AudioAsset("mary_had_lamb").get_local_path()
    with open(str(path), "rb") as f:
        audio_bytes = f.read()
    bio = io.BytesIO(audio_bytes)
    bio.name = "mary_had_lamb.wav"
    return bio


## Whisper (Encoder-Decoder) Tests
@pytest.fixture
def whisper_server():
    args = [
        "--dtype",
        "float16",
        "--enforce-eager",
    ]

    with RemoteOpenAIServer(WHISPER_MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def whisper_client(whisper_server):
    async with whisper_server.get_async_client() as async_client:
        yield async_client


# TODO - test multiple beams
@pytest.mark.asyncio
async def test_whisper_beam_search(
    whisper_client: openai.AsyncOpenAI,
    mary_had_lamb,
):
    """Test beam search with encoder-decoder model (Whisper) on transcriptions API."""
    mary_had_lamb.seek(0)

    transcription = await whisper_client.audio.transcriptions.create(
        model=WHISPER_MODEL_NAME,
        file=mary_had_lamb,
        language="en",
        response_format="text",
        temperature=0.0,
        extra_body=dict(
            use_beam_search=True,
        ),
    )

    result = json.loads(transcription)

    text = result["text"]

    assert text is not None
    assert len(text) > 0
    assert "mary had a little lamb" in text.lower()
