import torch
import torch.nn as nn
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from utils import *


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Initialize the linear transformation layers for key, value, query.
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        # This dropout is applied to normalized attention scores following the original
        # implementation of transformer. Although it is a bit unusual, we empirically
        # observe that it yields better performance.
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transform(self, x, linear_layer):
        # x is size: B, T, C
        # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
        bs, seq_len = x.shape[:2]
        proj = linear_layer(x)
        # Next, we need to produce multiple heads for the proj. This is done by spliting the
        # hidden state to self.num_attention_heads, each of size self.attention_head_size.
        proj = proj.view(bs, seq_len, self.num_attention_heads,
                         self.attention_head_size)
        # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
        proj = proj.transpose(1, 2)

        # Size: B, n, T, h
        return proj

    def attention(self, key, query, value, attention_mask):
        # Each attention is calculated following eq. (1) of https://arxiv.org/pdf/1706.03762.pdf.
        # Attention scores are calculated by multiplying the key and query to obtain
        # a score matrix S of size [bs, num_attention_heads, seq_len, seq_len].
        # S[*, i, j, k] represents the (unnormalized) attention score between the j-th and k-th
        # token, given by i-th attention head.
        # Before normalizing the scores, use the attention mask to mask out the padding token scores.
        # Note that the attention mask distinguishes between non-padding tokens (with a value of 0)
        # and padding tokens (with a value of a large negative number).

        # Make sure to:
        # - Normalize the scores with softmax.
        # - Multiply the attention scores with the value to get back weighted values.
        # - Before returning, concatenate multi-heads to recover the original shape:
        #   [bs, seq_len, num_attention_heads * attention_head_size = hidden_size].

        B, n, T, h = key.shape

        # Size: B, n, T, T
        wei = query @ key.transpose(-1, -2) * \
            (self.attention_head_size ** -0.5)

        # Attention_mask: [B, 1, 1, T], with 0 and -inf.
        # The following will be broadcasted.
        wei = wei + attention_mask
        wei = F.softmax(wei, dim=-1)

        # Size: B, n, T, h
        wei = wei @ value

        # Convert back to: B, T, C = n * h
        return wei.transpose(1, 2).contiguous().view(B, T, -1)

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: [bs, seq_len, hidden_state]
        attention_mask: [bs, 1, 1, seq_len]
        output: [bs, seq_len, hidden_state]
        """
        # First, we have to generate the key, value, query for each token for multi-head attention
        # using self.transform (more details inside the function).
        # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].

        # B, n, T, h
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)
        # Calculate the multi-head attention.
        attn_value = self.attention(
            key_layer, query_layer, value_layer, attention_mask)
        return attn_value


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Multi-head attention.
        self.self_attention = BertSelfAttention(config)
        # Add-norm for multi-head attention.
        self.attention_dense = nn.Linear(
            config.hidden_size, config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        # Feed forward.
        self.interm_dense = nn.Linear(
            config.hidden_size, config.intermediate_size)
        self.interm_af = F.gelu
        # Add-norm for feed forward.
        self.out_dense = nn.Linear(
            config.intermediate_size, config.hidden_size)
        self.out_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def add_norm(self, input, output, dense_layer, dropout, ln_layer):
        """
        This function is applied *after* the multi-head attention layer or the feed forward layer.
        input: the input of the previous layer, size B, T, C
        output: the output of the previous layer
        dense_layer: used to transform the output
        dropout: the dropout to be applied 
        ln_layer: the layer norm to be applied
        """
        # Hint: Remember that BERT applies dropout to the transformed output of each sub-layer,
        # before it is added to the sub-layer input and normalized with a layer norm.
        output = dense_layer(output)
        output = dropout(output)
        output = output + input
        return ln_layer(output)

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: either from the embedding layer (first BERT layer) or from the previous BERT layer
        as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf.
        Each block consists of:
        1. A multi-head attention layer (BertSelfAttention).
        2. An add-norm operation that takes the input and output of the multi-head attention layer.
        3. A feed forward layer.
        4. An add-norm operation that takes the input and output of the feed forward layer.
        """
        # hidden state size: B, T, C
        attn_out = self.self_attention(hidden_states, attention_mask)
        attn_out = self.add_norm(hidden_states, attn_out, self.attention_dense,
                                 self.attention_dropout, self.attention_layer_norm)
        dense_out = self.interm_af(self.interm_dense(attn_out))
        dense_out = self.add_norm(
            attn_out, dense_out, self.out_dense, self.out_dropout, self.out_layer_norm)

        # B, T, C
        return dense_out


class BertModel(BertPreTrainedModel):
    """
    The BERT model returns the final embeddings for each token in a sentence.

    The model consists of:
    1. Embedding layers (used in self.embed).
    2. A stack of n BERT layers (used in self.encode).
    3. A linear transformation layer for the [CLS] token (used in self.forward, as given).
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Embedding layers.
        self.word_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.pos_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.tk_type_embedding = nn.Embedding(
            config.type_vocab_size, config.hidden_size)
        self.embed_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
        # Register position_ids (1, len position emb) to buffer because it is a constant.
        position_ids = torch.arange(
            config.max_position_embeddings).unsqueeze(0)
        self.register_buffer('position_ids', position_ids)

        # BERT encoder.
        self.bert_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)])

        # [CLS] token transformations.
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()

        self.init_weights()

    def embed(self, input_ids):
        # B, T
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        # Get word embedding from self.word_embedding into input_embeds.
        input_embeds = self.word_embedding(input_ids)

        # Use pos_ids to get position embedding from self.pos_embedding into pos_embeds.
        # (1, T)
        pos_ids = self.position_ids[:, :seq_length]
        # (1, T, C)
        pos_embeds = self.pos_embedding(pos_ids)

        # Get token type ids. Since we are not considering token type, this embedding is
        # just a placeholder.
        tk_type_ids = torch.zeros(
            input_shape, dtype=torch.long, device=input_ids.device)
        tk_type_embeds = self.tk_type_embedding(tk_type_ids)

        # Add three embeddings together; then apply embed_layer_norm and dropout and return.
        embedding = input_embeds + pos_embeds + tk_type_embeds
        return self.embed_dropout(self.embed_layer_norm(embedding))

    def encode(self, hidden_states, attention_mask):
        """
        hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]
        """
        # Get the extended attention mask for self-attention.
        # Returns extended_attention_mask of size [batch_size, 1, 1, seq_len].
        # Distinguishes between non-padding tokens (with a value of 0) and padding tokens
        # (with a value of a large negative number).
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(
            attention_mask, self.dtype)

        # Pass the hidden states through the encoder layers.
        for i, layer_module in enumerate(self.bert_layers):
            # Feed the encoding from the last bert_layer to the next.
            hidden_states = layer_module(
                hidden_states, extended_attention_mask)

        # B, T, C
        return hidden_states

    def forward_with_embedding(self, embedding, attention_mask):
        """
        embedding: [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]
        """
        # Feed to a transformer (a stack of BertLayers).
        sequence_output = self.encode(
            embedding, attention_mask=attention_mask)

        # Get cls token hidden state.
        # Size: (B, C)
        first_tk = sequence_output[:, 0]
        first_tk = self.pooler_dense(first_tk)
        first_tk = self.pooler_af(first_tk)

        return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}

    def forward(self, ids_or_embeddings, attention_mask):
        """
        ids_or_embedding: [B, T], when the input is tokens. Or [B, T, C], where the input is embedding.
        attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
        """
        input_size = len(ids_or_embeddings.shape)
        if input_size != 2 and input_size != 3:
            raise ValueError(
                f"The BERT's input of unexpected shape: {ids_or_embeddings.shape}")

        # Get the embedding for each input token.
        if len(ids_or_embeddings.shape) == 2:
            ids_or_embeddings = self.embed(input_ids=ids_or_embeddings)

        # Feed the embedding to the transformer.
        return self.forward_with_embedding(ids_or_embeddings, attention_mask)
