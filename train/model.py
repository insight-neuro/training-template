import math

import numpy as np
import torch
import torch.nn as nn
from crane import BrainModel
from crane.core import BrainOutput
from omegaconf import DictConfig
from transformers import LlamaConfig, LlamaModel


class iEEGTransformer(BrainModel):
    """Transformer model for intracranial EEG (iEEG) data using LLaMA architecture with RoPE.

    Args:
        cfg: Configuration dictionary for the model.
        input_dim: Dimensionality of the input features (e.g., number of frequency bins).
    """

    def __init__(self, cfg: DictConfig, input_dim: int):
        max_position_embeddings = math.ceil(cfg.context_length / cfg.segment_length / (1 - cfg.p_overlap))
        rope_theta = max_position_embeddings * 2 * np.pi

        # LLaMA uses RoPE by default on the sequence dimension
        llama_config = LlamaConfig(
            hidden_size=cfg.transformer.d_model,
            num_hidden_layers=cfg.transformer.n_layers,
            num_attention_heads=cfg.transformer.n_heads,
            intermediate_size=4 * cfg.transformer.d_model,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            hidden_dropout_prob=cfg.dropout,
            attention_dropout=cfg.dropout,
        )

        super().__init__(llama_config)

        self.signal_projection = nn.Linear(input_dim, cfg.transformer.d_model)
        self.transformer = LlamaModel(llama_config)

        # Task-specific head (e.g., classification, regression, etc.)
        self.output_head = nn.Linear(cfg.transformer.d_model, input_dim)

    def forward(self, x: torch.Tensor) -> BrainOutput:
        """
        Args:
            x: [batch_size, num_electrodes, num_timesamples, num_frequencies]

        Returns:
            BrainOutput containing the last hidden state of the transformer.
        """
        batch_size, num_electrodes, num_timebins, _ = x.shape

        # Project signal values
        signal_emb = self.signal_projection(x)  # [batch_size, num_electrodes, num_timebins, d_model]
        position_ids = torch.arange(num_timebins, device=signal_emb.device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_electrodes, num_timebins)

        # Flatten to [B, N*M, d_model] - attend across all electrode-time pairs
        signal_emb = signal_emb.reshape(batch_size, num_electrodes * num_timebins, -1)  # [batch_size, num_electrodes * num_timebins, d_model]
        position_ids = position_ids.reshape(batch_size, num_electrodes * num_timebins)
        outputs = self.transformer(inputs_embeds=signal_emb, position_ids=position_ids)  # [batch_size, num_electrodes * num_timebins, d_model]

        # Get final representation
        hidden_states = outputs.last_hidden_state  # [batch_size, num_electrodes * num_timebins, d_model]
        hidden_states = hidden_states.reshape(batch_size, num_electrodes, num_timebins, -1)  # [batch_size, num_electrodes, num_timebins, d_model]

        return BrainOutput(last_hidden_state=hidden_states)
