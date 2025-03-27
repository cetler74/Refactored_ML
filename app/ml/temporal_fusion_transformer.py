# temporal_fusion_transformer.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


class TimeDistributed(nn.Module):
    """Applies a module over multiple time steps."""

    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        # Input shape: [batch, time steps, features]
        if x is None: # Handle optional inputs
             return None
        if x.ndim <= 2: # If input is static (batch, features), apply directly
             return self.module(x)

        batch_size, time_steps, features = x.size()

        # Reshape input to [batch * time steps, features]
        x_reshaped = x.contiguous().view(batch_size * time_steps, features)

        # Apply the module
        y = self.module(x_reshaped)

        # Reshape output back to [batch, time steps, output features]
        output_features = y.size(-1)
        y = y.contiguous().view(batch_size, time_steps, output_features)

        return y


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with interpretable weights."""

    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.1):
        super(InterpretableMultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads  # Dimension per head

        # Linear transformations for Q, K, V
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        # Output transformation
        self.output_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.d_k) # Precompute scale factor

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)

        # Reshape for multi-head attention
        # Shape: [batch_size, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Compute attention scores: Q * K^T / sqrt(d_k)
        # scores shape: [batch_size, num_heads, query_seq_len, key_seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask if provided (e.g., for decoder causality or padding)
        if mask is not None:
            # Ensure mask has compatible dimensions for broadcasting
            # Expected mask shape: [batch_size, 1, query_seq_len, key_seq_len] or similar
            scores = scores.masked_fill(mask == 0, -1e9) # Use large negative value

        # Apply softmax to get attention weights
        # attn_weights shape: [batch_size, num_heads, query_seq_len, key_seq_len]
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights) # Apply dropout to weights

        # Apply attention weights to values: weights * V
        # attn_output shape: [batch_size, num_heads, query_seq_len, d_k]
        attn_output = torch.matmul(attn_weights, V)

        # Reshape and concatenate heads
        # attn_output shape: [batch_size, query_seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final projection
        output = self.output_proj(attn_output)

        return output, attn_weights


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network as described in the TFT paper."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.1, context_dim: Optional[int] = None,
                 activation=nn.ELU): # Allow different activations
        super(GatedResidualNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        # Context transformation if provided
        if context_dim is not None:
            self.context_proj = nn.Linear(context_dim, hidden_dim)
        else:
            self.context_proj = None

        # Main network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = activation()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        # Gating layer - Input to gate should match input_dim
        self.gate_fc = nn.Linear(input_dim, output_dim) # Corrected: Gate depends on input 'x'
        self.gate_activation = nn.Sigmoid()

        # Skip connection projection if input and output dims differ
        if input_dim != output_dim:
            self.skip_proj = nn.Linear(input_dim, output_dim)
        else:
            self.skip_proj = None

        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x, context=None):
        # Calculate residual connection
        if self.skip_proj is not None:
            residual = self.skip_proj(x)
        else:
            residual = x # Direct skip connection if dims match

        # Main branch calculation
        hidden = self.fc1(x)
        if self.context_proj is not None and context is not None:
            context_transformed = self.context_proj(context)
            # Ensure context_transformed can be broadcast or added correctly
            # If context is per-batch and hidden is per-timestep, unsqueeze might be needed
            # Assuming hidden shape [batch, time, hidden_dim] and context [batch, hidden_dim]
            if hidden.ndim == 3 and context_transformed.ndim == 2:
                 context_transformed = context_transformed.unsqueeze(1) # Add time dimension
            hidden = hidden + context_transformed

        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden) # Output dim matches 'output_dim'

        # Gating mechanism
        gate = self.gate_activation(self.gate_fc(x)) # Gate calculated from original input 'x'

        # Apply gate: output = gate * main_path + (1 - gate) * residual_path
        gated_output = (gate * hidden) + ((1 - gate) * residual)

        # Layer normalization
        output = self.layer_norm(gated_output)

        return output


class VariableSelectionNetwork(nn.Module):
    """Variable selection network for TFT."""

    def __init__(self, input_dims: Dict[str, int], hidden_dim: int, output_dim: int,
                 dropout: float = 0.1, context_dim: Optional[int] = None,
                 presoftmax_output: bool = False): # Option to output before softmax
        super(VariableSelectionNetwork, self).__init__()
        if not input_dims:
            raise ValueError("input_dims dictionary cannot be empty.")
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim # This is the output dim of the *feature* GRNs
        self.num_inputs = len(input_dims)
        self.presoftmax_output = presoftmax_output

        # --- GRN for each input variable ---
        # Transforms each feature independently to 'output_dim'
        self.feature_grns = nn.ModuleDict()
        for name, dim in input_dims.items():
            # Each GRN outputs 'output_dim' features
            self.feature_grns[name] = GatedResidualNetwork(
                input_dim=dim,
                hidden_dim=hidden_dim, # Internal hidden dim
                output_dim=output_dim, # Output dim for the transformed feature
                dropout=dropout
            )

        # --- GRN for variable selection weights ---
        # Input is the concatenation of *original* input variables
        combined_input_dim = sum(input_dims.values())
        # Output dimension of selection GRN is the number of input variables
        self.selection_grn = GatedResidualNetwork(
            input_dim=combined_input_dim,
            hidden_dim=hidden_dim,
            output_dim=self.num_inputs, # Output one weight per input variable
            dropout=dropout,
            context_dim=context_dim
        )

        self.softmax = nn.Softmax(dim=-1) # Apply softmax across the variables dimension

    def forward(self, x_dict: Dict[str, torch.Tensor], context=None):
        # Ensure all expected inputs are present
        if not all(name in x_dict for name in self.input_dims):
             missing = set(self.input_dims) - set(x_dict)
             raise ValueError(f"Missing input features in x_dict: {missing}")

        # --- 1. Transform each variable independently ---
        # The transformed variables all have dimension 'output_dim'
        transformed_vars_list = []
        for name in self.input_dims.keys(): # Iterate in consistent order
            # Apply GRN; input shape [batch, (time), input_dim], output [batch, (time), output_dim]
            transformed_var = self.feature_grns[name](x_dict[name])
            transformed_vars_list.append(transformed_var)
        # Stack transformed variables: shape [batch, (time), num_inputs, output_dim]
        transformed_vars = torch.stack(transformed_vars_list, dim=-2)

        # --- 2. Compute variable selection weights ---
        # Concatenate original inputs: shape [batch, (time), combined_input_dim]
        combined_inputs = torch.cat([x_dict[name] for name in self.input_dims.keys()], dim=-1)

        # Pass concatenated inputs (and optional context) through selection GRN
        # selection_output shape: [batch, (time), num_inputs]
        selection_output = self.selection_grn(combined_inputs, context)

        if self.presoftmax_output:
             selection_weights_presoftmax = selection_output
             selection_weights = self.softmax(selection_output)
        else:
             selection_weights_presoftmax = None
             selection_weights = self.softmax(selection_output)

        # --- 3. Apply weights ---
        # Reshape weights for broadcasting: shape [batch, (time), num_inputs, 1]
        selection_weights_expanded = selection_weights.unsqueeze(-1)

        # Multiply weights with transformed variables element-wise
        weighted_vars = selection_weights_expanded * transformed_vars

        # Sum the weighted variables across the 'num_inputs' dimension
        # Output shape: [batch, (time), output_dim]
        combined_output = torch.sum(weighted_vars, dim=-2)

        # Return combined output and the weights (before softmax if requested)
        return combined_output, selection_weights_presoftmax if self.presoftmax_output else selection_weights


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer for interpretable time series forecasting."""

    def __init__(self,
                 # Feature dimension dictionaries
                 static_features: Dict[str, int],
                 time_varying_known_categoricals: Dict[str, int], # Renamed for clarity
                 time_varying_known_reals: Dict[str, int],      # Renamed for clarity
                 time_varying_observed_categoricals: Dict[str, int], # Renamed for clarity
                 time_varying_observed_reals: Dict[str, int],     # Renamed for clarity
                 # Model hyperparameters
                 hidden_dim: int = 64,
                 lstm_layers: int = 1, # Defaulted to 1 in paper/implementations
                 dropout: float = 0.1,
                 attn_heads: int = 4,
                 num_outputs: int = 1): # Usually 1 for regression/binary classification, or num_quantiles
        super(TemporalFusionTransformer, self).__init__()

        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.dropout_rate = dropout
        self.attn_heads = attn_heads
        self.num_outputs = num_outputs

        # --- Combine features for VSN input dimensions ---
        static_input_dims = static_features
        # Known features are used in both encoder and decoder
        known_cat_input_dims = time_varying_known_categoricals
        known_real_input_dims = time_varying_known_reals
        # Observed features only available in encoder
        observed_cat_input_dims = time_varying_observed_categoricals
        observed_real_input_dims = time_varying_observed_reals

        # --- 1. Input Transformations ---
        # Static Variable Selection and Context Generation
        if static_input_dims:
            self.static_vsn = VariableSelectionNetwork(
                input_dims=static_input_dims, hidden_dim=hidden_dim, output_dim=hidden_dim,
                dropout=dropout, presoftmax_output=True # Output weights before softmax for interpretability
            )
            # GRNs to generate static contexts
            self.static_context_variable_selection = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
            self.static_context_enrichment = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
            self.static_context_state_h = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
            self.static_context_state_c = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        else:
            self.static_vsn = None # And associated context GRNs are None

        # Time-Varying Input Variable Selection (applied per time step)
        # Combine all time-varying features for the selection GRN input
        # Encoder selection uses observed + known features
        encoder_all_dims = {**known_cat_input_dims, **known_real_input_dims,
                            **observed_cat_input_dims, **observed_real_input_dims}
        self.encoder_vsn = VariableSelectionNetwork(
            input_dims=encoder_all_dims, hidden_dim=hidden_dim, output_dim=hidden_dim,
            dropout=dropout, context_dim=hidden_dim if self.static_vsn else None,
            presoftmax_output=True
        )

        # Decoder selection uses only known features
        decoder_all_dims = {**known_cat_input_dims, **known_real_input_dims}
        self.decoder_vsn = VariableSelectionNetwork(
            input_dims=decoder_all_dims, hidden_dim=hidden_dim, output_dim=hidden_dim,
            dropout=dropout, context_dim=hidden_dim if self.static_vsn else None,
            presoftmax_output=True
        )

        # --- 2. Locality Enhancement with LSTM ---
        # Input to LSTM is the output of the VSN (hidden_dim)
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=lstm_layers, dropout=dropout if lstm_layers > 1 else 0, # Dropout only between LSTM layers
            batch_first=True
        )
        self.lstm_decoder = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=lstm_layers, dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True
        )

        # --- 3. Temporal Self-Attention ---
        # Applies attention across time steps after LSTM processing
        self.attention = InterpretableMultiHeadAttention(
            n_heads=attn_heads, d_model=hidden_dim, dropout=dropout
        )
        # Gated skip connection + LayerNorm after attention
        self.post_attention_gate = TimeDistributed(GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout))

        # --- 4. Position-wise Feed-Forward ---
        # Applied independently to each time step after attention
        self.pos_wise_ff_grn = TimeDistributed(GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout))

        # --- 5. Output Layer ---
        # Gated skip connection before final output projection
        self.pre_output_gate = TimeDistributed(GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout))
        # Final projection to the desired number of outputs (e.g., quantiles or single value)
        self.output_layer = TimeDistributed(nn.Linear(hidden_dim, num_outputs))

        # Store feature dims for reference
        self.static_dims = static_input_dims
        self.known_cat_dims = known_cat_input_dims
        self.known_real_dims = known_real_input_dims
        self.obs_cat_dims = observed_cat_input_dims
        self.obs_real_dims = observed_real_input_dims

    def forward(self, x: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the TFT model.

        Args:
            x: Dictionary containing input tensors. Expected keys:
                'static_inputs': Dict[str, Tensor] - Static features [batch, feat_dim]
                'encoder_known_categoricals': Dict[str, Tensor] - Known cats in encoder [batch, enc_len, feat_dim]
                'encoder_known_reals': Dict[str, Tensor] - Known reals in encoder [batch, enc_len, feat_dim]
                'encoder_observed_categoricals': Dict[str, Tensor] - Observed cats in encoder [batch, enc_len, feat_dim]
                'encoder_observed_reals': Dict[str, Tensor] - Observed reals in encoder [batch, enc_len, feat_dim]
                'decoder_known_categoricals': Dict[str, Tensor] - Known cats in decoder [batch, dec_len, feat_dim]
                'decoder_known_reals': Dict[str, Tensor] - Known reals in decoder [batch, dec_len, feat_dim]
                'encoder_lengths': Tensor (Optional) - Lengths for packing sequences
                'decoder_lengths': Tensor (Optional) - Lengths for packing sequences

        Returns:
            Tuple:
                - predictions: Output tensor [batch, decoder_length, num_outputs]
                - interpretability_weights: Dictionary containing attention weights and variable selection weights.
        """
        interpretability_weights = {}

        # --- Static Context Generation ---
        cs, ce, ch, cc = 0, 0, 0, 0 # Initialize static contexts to zero or learnable bias? Zero for now.
        if self.static_vsn is not None and x.get('static_inputs'):
            static_features = {k: x['static_inputs'][k] for k in self.static_dims}
            static_embedding, static_raw_weights = self.static_vsn(static_features)
            interpretability_weights["static_var_weights"] = self.static_vsn.softmax(static_raw_weights) # Save softmaxed weights
            cs = self.static_context_variable_selection(static_embedding)
            ce = self.static_context_enrichment(static_embedding)
            ch = self.static_context_state_h(static_embedding)
            cc = self.static_context_state_c(static_embedding)
        else:
            # If no static features, contexts remain zero (or could be learnable biases)
            pass

        # --- Encoder VSN ---
        # Combine all encoder inputs
        encoder_all_inputs = {}
        if x.get('encoder_known_categoricals'): encoder_all_inputs.update(x['encoder_known_categoricals'])
        if x.get('encoder_known_reals'): encoder_all_inputs.update(x['encoder_known_reals'])
        if x.get('encoder_observed_categoricals'): encoder_all_inputs.update(x['encoder_observed_categoricals'])
        if x.get('encoder_observed_reals'): encoder_all_inputs.update(x['encoder_observed_reals'])

        encoder_vsn_output, encoder_raw_weights = self.encoder_vsn(encoder_all_inputs, cs) # Pass static context cs
        interpretability_weights["encoder_var_weights"] = self.encoder_vsn.softmax(encoder_raw_weights)

        # --- Decoder VSN ---
        decoder_all_inputs = {}
        if x.get('decoder_known_categoricals'): decoder_all_inputs.update(x['decoder_known_categoricals'])
        if x.get('decoder_known_reals'): decoder_all_inputs.update(x['decoder_known_reals'])

        decoder_vsn_output, decoder_raw_weights = self.decoder_vsn(decoder_all_inputs, cs) # Pass static context cs
        interpretability_weights["decoder_var_weights"] = self.decoder_vsn.softmax(decoder_raw_weights)

        # --- LSTM Processing ---
        # Initialize LSTM states with static contexts (broadcast if necessary)
        # Assuming ch, cc shape [batch, hidden_dim]
        batch_size = encoder_vsn_output.size(0)
        # Reshape static contexts for LSTM: [num_layers, batch_size, hidden_dim]
        init_h = ch.unsqueeze(0).repeat(self.lstm_layers, 1, 1)
        init_c = cc.unsqueeze(0).repeat(self.lstm_layers, 1, 1)

        # LSTM Encoder
        # TODO: Handle packing/padding if using encoder_lengths
        encoder_lstm_output, (encoder_h, encoder_c) = self.lstm_encoder(encoder_vsn_output, (init_h, init_c))

        # LSTM Decoder
        # TODO: Handle packing/padding if using decoder_lengths
        decoder_lstm_output, _ = self.lstm_decoder(decoder_vsn_output, (encoder_h, encoder_c)) # Use encoder's final state

        # --- Static Enrichment ---
        # Add static context 'ce' to LSTM outputs
        # Need to potentially unsqueeze ce: [batch, 1, hidden_dim] for broadcasting
        static_enrichment = ce.unsqueeze(1) if ce.ndim == 2 else ce
        encoder_enriched = encoder_lstm_output + static_enrichment
        decoder_enriched = decoder_lstm_output + static_enrichment

        # --- Self-Attention Mechanism ---
        # Query: Decoder's enriched LSTM output
        # Key/Value: Concatenation of Encoder's and Decoder's enriched LSTM outputs (for full context)
        # Need to handle potential length differences if padding is used
        full_context_seq = torch.cat([encoder_enriched, decoder_enriched], dim=1)

        # Create attention mask if needed (e.g., decoder causality)
        # mask = create_decoder_mask(decoder_enriched.size(1)).to(decoder_enriched.device) # Example
        mask = None # Assuming no causality needed for now

        # Apply attention
        attn_output, attn_raw_weights = self.attention(
            query=decoder_enriched, # Query only uses decoder part
            key=full_context_seq,   # Key uses full sequence
            value=full_context_seq, # Value uses full sequence
            mask=mask
        )
        interpretability_weights["attention_weights"] = attn_raw_weights # Save raw attention weights

        # --- Post-Attention Gated Skip Connection ---
        # attn_output shape [batch, dec_len, hidden_dim]
        # decoder_enriched shape [batch, dec_len, hidden_dim]
        post_attn_output = self.post_attention_gate(attn_output + decoder_enriched) # Pass sum through gate

        # --- Position-wise Feed-forward ---
        # Applied to the output of the post-attention gate
        ff_output = self.pos_wise_ff_grn(post_attn_output)

        # --- Output Layer ---
        # Final gated skip connection + projection
        # ff_output shape [batch, dec_len, hidden_dim]
        # post_attn_output shape [batch, dec_len, hidden_dim]
        pre_output = self.pre_output_gate(ff_output + post_attn_output) # Pass sum through gate
        predictions = self.output_layer(pre_output) # Final linear projection

        # predictions shape [batch, dec_len, num_outputs]

        return predictions, interpretability_weights