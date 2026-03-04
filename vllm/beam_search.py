# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm.inputs import EncoderDecoderInputs, TokenInputs, token_inputs
from vllm.logprobs import Logprob
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalInputs, mm_inputs


@dataclass
class BeamSearchSequence:
    """A sequence for beam search.
    It keeps track of the tokens and the log probability of the sequence.
    The text field is optional and will only be filled when the sequence is
    about to be returned to the user.
    """

    orig_prompt: TokenInputs | MultiModalInputs | EncoderDecoderInputs

    # NOTE: Tokens represents decoder tokens in the encoder / decoder case
    tokens: list[int]
    logprobs: list[dict[int, Logprob]]
    lora_request: LoRARequest | None = None
    cum_logprob: float = 0.0
    text: str | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None

    def get_prompt(self):
        prompt = self.orig_prompt

        if prompt["type"] == "enc_dec":
            return self._build_encoder_decoder_inputs(prompt)

        # Handle decoder-only inputs
        prompt_text = prompt.get("prompt")
        cache_salt = prompt.get("cache_salt")

        if prompt["type"] == "token":
            return token_inputs(
                self.tokens,
                prompt=prompt_text,
                cache_salt=cache_salt,
            )

        return mm_inputs(
            prompt_token_ids=self.tokens,
            mm_kwargs=prompt["mm_kwargs"],
            mm_hashes=prompt["mm_hashes"],
            mm_placeholders=prompt["mm_placeholders"],
            prompt=prompt_text,
            cache_salt=cache_salt,
        )

    def _build_encoder_decoder_inputs(self, prompt):
        """Rebuild the encoder-decoder inputs with the current beam search
        sequence's tokens.

        NOTE: Because of the way that multimodal encoder/decoder caching is
        currently implemented, to prevent multimodal feature recomputation at
        decode time, we drop multimodal components from the decoder prompt.
        This is currently needed beacuse each beam runs as a new request
        generating one token, so num_computed_tokens=0, which is the condition
        for calling the encoder.

        TODO (alex) after caching handles this, update this accordingly.
        """
        dec_prompt = prompt["decoder_prompt"]

        self.orig_prompt: EncoderDecoderInputs
        num_current_tokens = len(self.tokens)
        num_initial_tokens = len(self.orig_prompt["decoder_prompt"]["prompt_token_ids"])
        is_decode = num_current_tokens > num_initial_tokens

        if not is_decode and dec_prompt["type"] == "multimodal":
            mm_decoder_info = {
                "mm_kwargs": dec_prompt["mm_kwargs"],
                "mm_hashes": dec_prompt["mm_hashes"],
                "mm_placeholders": dec_prompt["mm_placeholders"],
            }
        else:
            mm_decoder_info = {
                "mm_kwargs": None,
                "mm_hashes": {},
                "mm_placeholders": {},
            }

        # Rebuild decoder prompt with updated tokens,
        # but keep everything else the same.
        new_dec_prompt = mm_inputs(
            self.tokens,
            prompt=dec_prompt.get("prompt"),
            cache_salt=dec_prompt.get("cache_salt"),
            **mm_decoder_info,
        )

        return EncoderDecoderInputs(
            type="enc_dec",
            encoder_prompt=prompt["encoder_prompt"],
            decoder_prompt=new_dec_prompt,
        )


@dataclass
class BeamSearchOutput:
    """The output of beam search.
    It contains the list of the best beam search sequences.
    The length of the list is equal to the beam width.
    """

    sequences: list[BeamSearchSequence]


class BeamSearchInstance:
    def __init__(
        self,
        prompt: TokenInputs | MultiModalInputs | EncoderDecoderInputs,
        lora_request: LoRARequest | None = None,
        logprobs: list[dict[int, Logprob]] | None = None,
        **kwargs,
    ):
        decoder_prompt = (
            prompt if prompt["type"] != "enc_dec" else prompt["decoder_prompt"]
        )
        initial_tokens = decoder_prompt["prompt_token_ids"]

        self.beams: list[BeamSearchSequence] = [
            BeamSearchSequence(
                orig_prompt=prompt,
                tokens=initial_tokens,
                logprobs=[] if logprobs is None else list(logprobs),
                lora_request=lora_request,
                **kwargs,
            )
        ]
        self.completed: list[BeamSearchSequence] = []


def get_beam_search_score(
    tokens: list[int],
    cumulative_logprob: float,
    eos_token_id: int,
    length_penalty: float = 1.0,
) -> float:
    """Calculate the beam search score with length penalty.

    Adapted from

    https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L938
    """
    seq_len = len(tokens)
    if tokens[-1] == eos_token_id:
        seq_len -= 1

    return cumulative_logprob / (seq_len**length_penalty)


def create_sort_beams_key_function(eos_token_id: int, length_penalty: float):
    def sort_beams_key(x: BeamSearchSequence) -> float:
        return get_beam_search_score(
            x.tokens, x.cum_logprob, eos_token_id, length_penalty
        )

    return sort_beams_key
