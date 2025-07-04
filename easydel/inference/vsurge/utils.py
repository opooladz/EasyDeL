# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import typing as tp
from bisect import bisect_left

import jax
import numpy as np
from jax import numpy as jnp

from .vengine import ResultTokens

if tp.TYPE_CHECKING:
	from easydel.infra.utils import ProcessingClassType
else:
	ProcessingClassType = tp.Any


@dataclasses.dataclass
class ReturnSample:
	"""Represents a single generated sample with text, token IDs, and metrics.

	This dataclass encapsulates the output for one sample (sequence) from a
	generation step, including the detokenized text, the raw token IDs, and
	performance metrics like tokens per second and the cumulative number of
	generated tokens.

	Attributes:
	  text: A list of string pieces detokenized from the token IDs. This can be
	        a single string or a list of strings if dealing with byte tokens
	        or streaming output.
	  token_ids: A list of integer token IDs generated in this step.
	  tokens_per_second: The cumulative tokens per second achieved for this sample
	                     up to the current generation step. Optional.
	  num_generated_tokens: The cumulative number of tokens generated for this
	                        sample since the start of the decode phase. Optional.
	"""

	text: list[str]
	token_ids: list[int]
	tokens_per_second: float | None = dataclasses.field(default=None)
	num_generated_tokens: int | None = dataclasses.field(default=None)


def process_result_tokens(
	processor: ProcessingClassType,
	slot: int,
	slot_max_length: int,
	result_tokens: ResultTokens,
	complete: np.ndarray,
	eos_token_id: list[int],
	is_client_side_tokenization: bool = False,
) -> tp.Tuple[tp.List[ReturnSample], np.ndarray, list[int]]:
	"""
	Processes the result tokens for a given slot, extracts text and token IDs,
	updates completion status, and counts valid tokens generated in this step.

	Args:
	    processor: The tokenizer/processor instance.
	    slot: The index of the inference slot being processed.
	    slot_max_length: The maximum allowed length for the sequence in this slot.
	    result_tokens: The ResultTokens object containing the generated tokens and metadata.
	    complete: A boolean NumPy array indicating the completion status of each sample in the batch.
	    is_client_side_tokenization: A boolean indicating if tokenization is handled client-side.

	Returns:
	    A tuple containing:
	        - A list of ReturnSample objects (without TPS/count populated yet).
	        - The updated completion status array.
	        - A list containing the number of valid tokens generated in this step
	          for each corresponding ReturnSample.
	"""
	slot_data = result_tokens.get_result_at_slot(slot)
	slot_tokens = slot_data.tokens
	slot_valid = slot_data.valid
	slot_lengths = slot_data.lengths
	samples, speculations = slot_tokens.shape

	if isinstance(eos_token_id, int):
		eos_token_id = [eos_token_id]
	complete = complete | (slot_lengths > slot_max_length)
	return_samples = []
	num_valid_tokens_step = []  # Track valid tokens generated in this step per sample
	for idx in range(samples):
		text_so_far = []
		tok_id_so_far = []
		valid_tokens_count = 0
		if not complete[idx].item():
			for spec_idx in range(speculations):
				tok_id = slot_tokens[idx, spec_idx].item()
				valid = slot_valid[idx, spec_idx].item()
				if tok_id in eos_token_id or not valid:
					complete[idx] = True
					# Include EOS token in count if valid
					if valid and tok_id in eos_token_id:
						tok_id_so_far.append(tok_id)
						valid_tokens_count += 1
					break
				else:
					if not is_client_side_tokenization:
						text_so_far.append(processor.decode([tok_id]))
					tok_id_so_far.append(tok_id)
					valid_tokens_count += 1
		# Append a base ReturnSample without TPS/count yet
		return_samples.append(ReturnSample(text=text_so_far, token_ids=tok_id_so_far))
		num_valid_tokens_step.append(valid_tokens_count)
	return return_samples, complete, num_valid_tokens_step


def tokenize_and_pad(
	string: str,
	processor: ProcessingClassType,
	is_bos: bool = True,
	prefill_lengths: tp.Optional[tp.List[int]] = None,
	max_prefill_length: tp.Optional[int] = None,
	jax_padding: bool = True,
) -> tp.Tuple[tp.Union[jax.Array, np.ndarray], tp.Union[jax.Array, np.ndarray], int]:
	"""Tokenizes an input string and pads it to a suitable length.

	Uses the provided processor to tokenize the input string, then pads the
	resulting token IDs and attention mask (valids) to the nearest length
	specified in `prefill_lengths` or up to `max_prefill_length`. Optionally
	prepends the BOS token.

	Args:
	    string: The input string to tokenize.
	    processor: The tokenizer/processor object.
	    is_bos: Whether to prepend the beginning-of-sequence (BOS) token.
	        Defaults to True. (Note: BOS handling seems missing in the
	        current `pad_tokens` implementation called internally).
	    prefill_lengths: A list of bucket lengths to pad to. If None, uses
	        `DEFAULT_PREFILL_BUCKETS`.
	    max_prefill_length: The maximum allowed prefill length. Overrides
	        buckets larger than this value.
	    jax_padding: If True, returns JAX arrays; otherwise, returns NumPy arrays.
	        Defaults to True.

	Returns:
	    A tuple containing:
	        - padded_tokens: The padded token ID array (JAX or NumPy).
	        - padded_valids: The padded attention mask array (JAX or NumPy).
	        - padded_length: The length to which the arrays were padded/truncated.
	"""
	content = processor(
		string,
		return_tensors="np",
		return_attention_mask=True,
	)
	tokens = np.array(content["input_ids"])
	valids = np.array(content["attention_mask"])
	bos_token_id = processor.bos_token_id
	pad_token_id = processor.pad_token_id

	padded_tokens, padded_valids, padded_length = pad_tokens(
		tokens=tokens,
		valids=valids,
		bos_token_id=bos_token_id,
		pad_token_id=pad_token_id,
		is_bos=is_bos,
		prefill_lengths=prefill_lengths,
		max_prefill_length=max_prefill_length,
		jax_padding=jax_padding,
	)
	return padded_tokens, padded_valids, padded_length


DEFAULT_PREFILL_BUCKETS = [2**s for s in range(5, 24)]


def take_nearest_length(lengths: list[int], length: int) -> int:
	"""Gets the nearest length to the right in a set of lengths.

	Uses binary search to find the smallest length in the `lengths` list that is
	greater than or equal to the input `length`.

	Args:
	    lengths: A sorted list of integer lengths (e.g., prefill buckets).
	    length: The target length to find the nearest value for.

	Returns:
	    The nearest length in `lengths` that is greater than or equal to `length`.
	    If `length` is greater than all lengths in the list, returns the largest length.
	"""
	pos = bisect_left(lengths, length)
	if pos == len(lengths):
		return lengths[-1]
	return lengths[pos]


def pad_tokens(
	tokens: np.ndarray,
	valids: np.ndarray,
	pad_token_id: int,
	prefill_lengths: tp.Optional[tp.List[int]] = None,
	max_prefill_length: tp.Optional[int] = None,
	jax_padding: bool = True,
	right_padding: bool = False,
	bos_token_id: int | None = None,  # Added for clarity, though not used
	is_bos: bool = True,  # Added for clarity, though not used
) -> tp.Tuple[tp.Union[jax.Array, np.ndarray], tp.Union[jax.Array, np.ndarray], int]:
	"""Pads token and validity arrays to a specified bucket length.

	Takes 1D NumPy arrays of token IDs and validity masks, determines the
	nearest appropriate padding length from `prefill_lengths` (or capped by
	`max_prefill_length`), and pads or truncates the arrays to that length.
	Padding uses the `pad_token_id` for tokens and 0 for validity.

	Note: The `bos_token_id` and `is_bos` arguments are included for potential
	future use or consistency with `tokenize_and_pad`, but they are not
	currently used within this function's logic. BOS token handling should
	be done before calling this function if required.

	Args:
	    tokens: A 1D NumPy array of token IDs.
	    valids: A 1D NumPy array representing the attention mask (1 for valid,
	        0 for padding). Must be the same size as `tokens`.
	    pad_token_id: The token ID used for padding.
	    prefill_lengths: A list of integer bucket lengths to choose from.
	        Defaults to `DEFAULT_PREFILL_BUCKETS`.
	    max_prefill_length: An optional maximum length. If provided, buckets
	        larger than this are ignored, and this value is used as the maximum
	        padding length.
	    jax_padding: If True, converts the padded NumPy arrays to JAX arrays
	        before returning. Defaults to True.
	    bos_token_id: The beginning-of-sequence token ID (currently unused).
	    is_bos: Flag indicating if BOS token handling is expected (currently unused).

	Returns:
	    A tuple containing:
	        - padded_tokens: The padded/truncated token ID array (JAX or NumPy).
	        - padded_valids: The padded/truncated validity mask array (JAX or NumPy).
	        - padded_length: The length to which the arrays were padded/truncated.
	"""
	if prefill_lengths is None:
		prefill_lengths = DEFAULT_PREFILL_BUCKETS
	if max_prefill_length is not None:
		prefill_lengths = prefill_lengths[: prefill_lengths.index(max_prefill_length)] + [
			max_prefill_length
		]
	tokens = tokens.ravel()  # 1d Only
	valids = valids.ravel()
	true_length = tokens.shape[-1]
	assert valids.size == tokens.size
	padded_length = take_nearest_length(prefill_lengths, true_length)
	padding = padded_length - true_length
	if padding < 0:
		padded_tokens = tokens[-padded_length:]
		padded_valids = valids[-padded_length:]
	else:
		paddin = (0, padding) if right_padding else (padding, 0)
		padded_tokens = np.pad(tokens, paddin, constant_values=(pad_token_id,))
		padded_valids = np.pad(valids, paddin, constant_values=(0,))
	if jax_padding:
		padded_tokens = jnp.array([padded_tokens])
		padded_valids = jnp.array([padded_valids])
	return padded_tokens, padded_valids, true_length


def is_byte_token(s: str) -> bool:
	"""Returns True if s is a byte string like "<0xAB>".

	These tokens represent raw bytes and are used in some tokenization schemes
	to handle multi-byte characters or special symbols.

	Args:
	    s: The input string to check.

	Returns:
	    True if the string matches the byte token format "<0xXX>", False otherwise.
	"""
	if len(s) != 6 or s[0:3] != "<0x" or s[-1] != ">":
		return False
	return True


def text_tokens_to_string(text_tokens: tp.Iterable[str]) -> str:
	"""Converts an iterable of text tokens, including byte tokens, to a string.

	This function handles tokens that represent raw bytes (e.g., "<0xAB>")
	correctly by converting them to their byte values before decoding the
	entire sequence of bytes into a UTF-8 string. This is necessary for
	tokenizers that output byte tokens for special characters or multi-byte
	sequences.

	Iterates through text tokens. If a token represents a byte (e.g., "<0xAB>"),
	it's converted to its byte value. Otherwise, the token is treated as a
	UTF-8 string and converted to bytes. All resulting bytes are joined and
	decoded back into a single UTF-8 string, replacing errors.

	Args:
	    text_tokens: An iterable (e.g., list) of string tokens, which may include
	                 byte tokens in the format "<0xXX>".

	Returns:
	    The decoded string representation of the token sequence.
	"""
	bytes_so_far = []
	for text_token in text_tokens:
		if is_byte_token(text_token):
			bytes_so_far.append(bytes([int(text_token[1:-1], 16)]))
		else:
			bytes_so_far.append(bytes(text_token, "utf-8"))
	return b"".join(bytes_so_far).decode("utf-8", "replace")
