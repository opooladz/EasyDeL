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

import typing as tp
from functools import cached_property

import jax
from eformer import escale as es
from flax import nnx as nn
from jax import numpy as jnp
from jax.sharding import NamedSharding as Ns
from jax.sharding import PartitionSpec as Ps
from transformers import ProcessorMixin
from eformer import common_types
from easydel.utils.compiling_utils import cjit

from .utilities import (
	GenerationState,
	ResultTokens,
	continuous_bulk_insert,
	continuous_decode,
	continuous_insert,
	continuous_prefill,
)

if tp.TYPE_CHECKING:
	from easydel.infra import EasyDeLBaseModule
else:
	EasyDeLBaseModule = tp.Any


import jax.experimental.layout as jlu

AutoLayout = jlu.Layout(jlu.DeviceLocalLayout.AUTO)

NOT_GIVEN = common_types.NOT_GIVEN
RUNTIME_MODE_TYPES = common_types.RUNTIME_MODE_TYPES
BATCH = common_types.BATCH
QUERY_LENGTH = common_types.QUERY_LENGTH
KV_LENGTH = common_types.KV_LENGTH
HEAD = common_types.HEAD
KV_HEAD = common_types.KV_HEAD
HEAD_DIM = common_types.HEAD_DIM
KV_HEAD_DIM = common_types.KV_HEAD_DIM
BIAS_HEAD_SEQ = common_types.BIAS_HEAD_SEQ
BIAS_KV_SEQ = common_types.BIAS_KV_SEQ
MODE_PREFILL = common_types.MODE_PREFILL


class vEngine:
	"""
	Core inference engine for EasyDeL models using NNX graphs.

	This engine manages the model state (split into graph definition, state, and
	other parameters) and provides JIT-compiled functions for the prefill and
	decode steps of autoregressive generation. It handles KV caching and
	sampling.
	"""

	def __init__(
		self,
		model: EasyDeLBaseModule,
		processor,
		max_concurrent_decodes: int | None = None,
		max_concurrent_prefill: int | None = None,
		prefill_lengths: int | None = None,
		max_prefill_length: int | None = None,
		max_length: int | None = None,
		batch_size: int | None = None,
		seed: int = 894,
	):
		"""Initializes the vEngine.

		Args:
		    model: The EasyDeLBaseModule (NNX) to use for inference.
		    processor: The tokenizer/processor associated with the model.
		    max_concurrent_decodes: The maximum number of sequences that can be
		        decoded concurrently. Defaults to the number of available JAX devices.
		    max_concurrent_prefill: The maximum number of prefill requests that can be
		        processed concurrently. Defaults to the number of available JAX devices.
		    prefill_lengths: A list of integer bucket lengths to choose from for prefill.
		        Defaults to `DEFAULT_PREFILL_BUCKETS`.
		    max_prefill_length: The maximum allowed length for the initial prompt
		        (prefill phase). Defaults to 4096.
		    max_length: The maximum total sequence length (prompt + generation).
		        Defaults to 8192.
		    batch_size: The batch size for the engine. Defaults to None.
		    seed: The random seed for initializing the PRNG key used in sampling.
		        Defaults to 894.
		"""
		from ..utils import DEFAULT_PREFILL_BUCKETS

		self.model = model
		self.graphdef, self.graphstate, self.graphothers = model.split_module()
		self.processor = processor
		self._quantizer = model._quant_class(
			quantization_method=model.config.kv_cache_quantization_method,
			block_size=model.config.kv_cache_quantization_blocksize,
			quantization_platform=model.config.platform,
		)
		self._partition_manager = model.config.partition_manager
		self._max_prefill_lengths = prefill_lengths or DEFAULT_PREFILL_BUCKETS
		self._max_concurrent_decodes = max_concurrent_decodes or jax.device_count()
		self._max_concurrent_prefill = max_concurrent_prefill or jax.device_count()
		self._max_prefill_length = max_prefill_length or 4096
		self._max_length = max_length or 8192
		self._batch_size = batch_size
		self._prng_key = jax.random.PRNGKey(seed)
		self._empty_sharding = Ns(model.mesh, Ps())
		self._setup_functions()

	def get_state_shardings(self, is_decode: bool = False):
		pmang = self.model.config.partition_manager
		kvps = pmang.resolve(
			axes=[BATCH if is_decode else "_", KV_LENGTH, KV_HEAD, KV_HEAD_DIM],
			mode=MODE_PREFILL,
		)
		bsharding = pmang.resolve(axes=[BATCH if is_decode else "_"], mode=MODE_PREFILL)

		return (
			("index", bsharding),
			("logits", bsharding),
			("tokens", bsharding),
			("valids", bsharding),
			("next_position_ids", bsharding),
			("temperature", bsharding),
			("top_p", bsharding),
			("top_k", bsharding),
			("generated_tokens", bsharding),
			("cache/views/[0-9]+/(key|value)/(scale|weight)", kvps),
			("cache/views/[0-9]+/(key|value)/(packed|absmax)", kvps),
			("cache/views/[0-9]+/(key|value)", kvps),
			("cache/views/[0-9]+/index", bsharding),
			("cache/views/[0-9]+/starts", bsharding),
			(".*", Ps()),
		)

	def _setup_functions(self):
		with self.model.mesh:
			self._prefill_state_sharding = jax.tree_util.tree_map(
				lambda x: Ns(self.model.mesh, x),
				es.match_partition_rules(
					self.get_state_shardings(False),
					jax.eval_shape(self.init_decode_state),
					min_size=0,
				),
			)
			self._decodes_state_sharding = jax.tree_util.tree_map(
				lambda x: Ns(self.model.mesh, x),
				es.match_partition_rules(
					self.get_state_shardings(True),
					jax.eval_shape(self.init_decode_state),
					min_size=0,
				),
			)

			self.continuous_prefill = cjit(
				jax.jit(
					continuous_prefill,
					static_argnums=(0, 10),
					in_shardings=(
						es.extract_shardings(self.graphstate),
						es.extract_shardings(self.graphothers),
						self._empty_sharding,
						self._empty_sharding,
						None,
						None,
						self._empty_sharding,
						self._empty_sharding,
						self._empty_sharding,
						None,
						None,
					),
					out_shardings=(self._prefill_state_sharding, None),
				),
				static_argnums=(0, 10),
			)
			self.continuous_decode = cjit(
				jax.jit(
					continuous_decode,
					static_argnums=(0,),
					donate_argnums=(3,),
					in_shardings=(
						es.extract_shardings(self.graphstate),
						es.extract_shardings(self.graphothers),
						self._decodes_state_sharding,
						None,
						None,
					),
					out_shardings=(self._decodes_state_sharding, None),
				),
				static_argnums=(0,),
			)
			self.continuous_insert = cjit(
				jax.jit(
					continuous_insert,
					donate_argnums=(0, 1),
					static_argnums=(3, 4),
					in_shardings=(
						self._prefill_state_sharding,
						self._decodes_state_sharding,
						None,
					),
					out_shardings=self._decodes_state_sharding,
				),
				static_argnums=(3, 4),
			)

	@cached_property
	def pad_token_id(self):
		if isinstance(self.processor, ProcessorMixin):
			pad_token_id = self.processor.tokenizer.pad_token_id
		else:
			pad_token_id = self.processor.pad_token_id
		return pad_token_id

	@cached_property
	def eos_token_ids(self) -> list[int]:
		eos_ids = []
		if isinstance(self.processor, ProcessorMixin):
			proc_eos_token_id = self.processor.tokenizer.eos_token_id
		else:
			proc_eos_token_id = self.processor.eos_token_id
		if isinstance(proc_eos_token_id, int):
			proc_eos_token_id = [proc_eos_token_id]
		eos_ids = eos_ids + proc_eos_token_id
		if hasattr(self.model, "generation_config"):
			conf_eos = self.model.generation_config.eos_token_id
			if isinstance(conf_eos, int):
				conf_eos = [conf_eos]
			eos_ids = eos_ids + conf_eos
		return list(set(eos_ids))

	@property
	def samples_per_slot(self) -> int:
		"""Number of samples generated per inference slot.

		This determines how many independent generation results are produced
		for each logical slot managed by the engine. It's often 1, but could
		be higher for techniques like parallel sampling.
		"""
		return 1  # self._max_concurrent_decodes // self._max_concurrent_prefill

	@property
	def prng_key(self) -> jax.random.PRNGKey:
		"""Provides a new PRNG key split from the internal state for sampling.

		Each call to this property consumes the current key and returns a new,
		unique key, ensuring that subsequent sampling operations use different
		randomness.

		Returns:
		    A new JAX PRNGKey.
		"""
		self._prng_key, new_key = jax.random.split(self._prng_key, 2)
		return new_key

	@property
	def prefill_lengths(self) -> list[int]:
		"""Returns the configured list of max prefill length buckets for the engine."""
		return self._max_prefill_lengths

	@property
	def batch_size(self) -> int | None:
		"""Returns the configured batch size for the engine, if specified."""
		return self._batch_size

	@property
	def max_concurrent_decodes(self) -> int:
		"""Maximum number of sequences that can be decoded concurrently.

		This determines the batch size used during the decode phase.
		"""
		return self._max_concurrent_decodes

	@property
	def max_prefill_length(self) -> int:
		"""Maximum allowed length for the initial prompt (prefill phase).

		Prompts longer than this will be truncated or handled according to
		the padding/truncation logic.
		"""
		return self._max_prefill_length

	@property
	def max_length(self) -> int:
		"""Maximum total sequence length (prompt + generation).

		This defines the size of the KV cache allocated.
		"""
		return self._max_length

	def get_prefix_destination_sharding(self) -> tp.Any:
		"""Returns the shardings necessary to transfer KV cache data between engines.

		Currently returns None, indicating default or no specific sharding.
		"""
		return self._prefill_state_sharding

	def init_decode_state(self, *args, **kwargs) -> GenerationState:
		"""Initializes the GenerationState for a new sequence"""
		concurrent_decode = self.max_concurrent_decodes
		pmang = self.model.config.partition_manager

		sharding = Ns(self.model.mesh, pmang.resolve(axes=[BATCH], mode=MODE_PREFILL))

		vocab_size = getattr(self.model.config, "vocab_size", None)
		if vocab_size is None and hasattr(self.model.config, "text_config"):
			vocab_size = getattr(self.model.config.text_config, "vocab_size", None)
		assert vocab_size is not None, "couldn't find `vocab_size` in model config"
		dt = self.model.dtype
		with self.model.mesh:
			return GenerationState(
				logits=jnp.zeros((concurrent_decode, vocab_size), dt, device=sharding),
				cache=self.model.init_cache(concurrent_decode, self.max_length),
				index=jnp.zeros((concurrent_decode, 1), "i4", device=sharding),
				tokens=jnp.zeros((concurrent_decode, 1), "i4", device=sharding),
				valids=jnp.zeros((concurrent_decode, self.max_length), "b1", device=sharding),
				next_position_ids=jnp.zeros((concurrent_decode, 1), "i4", device=sharding),
				temperature=jnp.zeros((concurrent_decode,), "f4", device=sharding),
				top_p=jnp.zeros((concurrent_decode,), "f4", device=sharding),
				top_k=jnp.zeros((concurrent_decode,), "i4", device=sharding),
				generated_tokens=jnp.zeros((concurrent_decode, 1), "i4", device=sharding),
			)

	def free_resource(self, slot: int) -> bool:
		"""Frees resources associated with a specific inference slot. (Not Implemented)

		Args:
		    slot: The index of the slot to free.

		Returns:
		    Always returns False as it's not implemented.
		"""
		return False  # Placeholder: Implementation needed

	@property
	def colocated_cpus(self) -> tp.Union[list[jax.Device], None]:
		"""Returns CPU devices colocated with the engine's accelerator devices.

		This information can be useful for optimizing data transfers between
		host (CPU) and accelerator (GPU/TPU) memory. Currently returns None
		as the implementation is pending.

		Returns:
		    A list of colocated JAX CPU devices, or None if not implemented or available.
		"""
		return None  # Placeholder: Implementation needed

	def prefill(
		self,
		graphstate: nn.GraphState,
		graphothers: nn.GraphState,
		tokens: jax.Array,
		valids: jax.Array,
		true_length: int,
		temperature: jax.Array,
		top_p: jax.Array,
		top_k: jax.Array,
		rngs: jax.random.PRNGKey,
	) -> tuple[GenerationState, ResultTokens]:
		"""Performs the prefill step for initializing the generation process.

		Processes the initial prompt tokens, initializes the KV cache, and generates
		the *first* token of the sequence. This function is JIT-compiled.

		Args:
		    graphstate: The NNX GraphState (parameters) of the model.
		    graphothers: Other NNX state variables of the model.
		    tokens: The input prompt token IDs (batch_size, sequence_length).
		    valids: A boolean array indicating valid token positions in the input
		        (batch_size, sequence_length or batch_size, max_length).
		    rngs: A JAX PRNG key for sampling the first token.

		Returns:
		    A tuple containing:
		        - generation_state: The initial GenerationState for the decode loop.
		        - result: A ResultTokens object containing the *first* generated token.
		"""
		return self.continuous_prefill(
			self.graphdef,
			graphstate,
			graphothers,
			tokens,
			valids,
			true_length,
			self.pad_token_id,
			temperature,
			top_p,
			top_k,
			self.max_length,
			self.samples_per_slot,
			rngs,
		)

	def decode(
		self,
		graphstate: nn.GraphState,
		graphothers: nn.GraphState,
		state: GenerationState,
		rngs: jax.random.PRNGKey,
	) -> tuple[GenerationState, ResultTokens]:
		"""Performs a single decode step in the autoregressive generation loop.

		Takes the previous GenerationState, generates the next token using the model
		and KV cache, and updates the state. This function is JIT-compiled and
		allows the input state's cache to be modified in-place (donated).

		Args:
		    graphstate: The NNX GraphState (parameters) of the model.
		    graphothers: Other NNX state variables of the model.
		    state: The current GenerationState from the previous step. `state.cache`
		        is marked for donation.
		    rngs: A JAX PRNG key for sampling the next token.

		Returns:
		    A tuple containing:
		        - next_generation_state: The updated GenerationState for the next iteration.
		        - result: A ResultTokens object containing the newly generated token.
		"""
		return self.continuous_decode(
			self.graphdef,
			graphstate,
			graphothers,
			state,
			self.samples_per_slot,
			rngs,
		)

	def insert(
		self,
		prefix: GenerationState,
		decode_state: GenerationState,
		slot: int,
	) -> GenerationState:
		"""Inserts or updates a generation state, potentially for managing batches. (JIT-compiled)

		This function seems designed to merge or update parts of the generation state,
		possibly inserting a 'prefix' state (e.g., from a completed prefill) into
		a larger batch state ('decode_state') at a specific 'slot'. The exact
		mechanism for insertion isn't fully clear from the current implementation,
		as it primarily focuses on broadcasting the prefix cache and returning the
		prefix state. Both input states' caches are donated.

		Args:
		    prefix: The GenerationState to potentially insert (e.g., from prefill).
		        Its cache is marked for donation.
		    decode_state: The target GenerationState to update (e.g., the main decode loop state).
		        Its cache is marked for donation.
		    slot: The index within the batch where the insertion/update should occur.

		Returns:
		    An updated GenerationState. In the current implementation, it returns the
		    prefix state with its cache potentially broadcasted. Needs clarification
		    on the intended merging logic with `decode_state` and `slot`.
		"""
		with self.model.mesh:
			return self.continuous_insert(
				prefix,
				decode_state,
				slot,
				self._quantizer,
				self._partition_manager,
			)

	def bulk_insert(
		self,
		prefix: GenerationState,
		decode_state: GenerationState,
		slots: list[int],
	) -> GenerationState:
		"""Efficiently inserts multiple prefill results into the decode state.

		This function takes a `GenerationState` (`prefix`) typically resulting
		from a batch prefill operation and inserts its relevant components
		(logits, cache, index, tokens, valids, position IDs, generated tokens)
		into the main `decode_state` at multiple specified `slots`. This is
		useful for initializing the decode state after processing a batch of
		prompts simultaneously. Both input states' caches are donated.

		Args:
		    prefix: The `GenerationState` containing the results from a prefill
		        operation (or similar initialization). Its cache is marked for
		        donation.
		    decode_state: The target `GenerationState` (e.g., the main decode
		        loop state) to be updated. Its cache is marked for donation.
		    slots: A list of integer indices indicating the slots within the
		        `decode_state`'s batch dimension where the corresponding data
		        from the `prefix` state should be inserted.

		Returns:
		    An updated `GenerationState` (`decode_state`) with the prefill
		    results inserted at the specified slots.
		"""

		with self.model.mesh:
			return continuous_bulk_insert(
				prefix=prefix,
				decode_state=decode_state,
				slots=slots,
				quantizer=self._quantizer,
				partition_manager=self._partition_manager,
			)
