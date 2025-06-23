import mlx.core as mx
import mlx.nn as nn
import mlx.utils as utils
import math

### hyper params
# model
ctx_len = 128
n_emb = 128
dropout = 0.1
head_size = 128
n_heads = 4 
n_layers = 3

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = nn.Linear(n_emb, head_size, bias=False)
        self.q_proj = nn.Linear(n_emb, head_size, bias=False)
        self.v_proj = nn.Linear(n_emb, head_size, bias=False)
        indices = mx.arange(ctx_len)
        mask = indices[:, None] < indices[None]
        self._causal_mask = mask * -1e9
        self.c_proj = nn.Linear(head_size, n_emb)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
    def __call__(self, x):
        B, T, C = x.shape
        K = self.k_proj(x)
        Q = self.q_proj(x)
        V = self.v_proj(x)
        mha_shape = (B, T, n_heads, head_size//n_heads)
        K = mx.as_strided(K, (mha_shape)).transpose([0, 2, 1, 3])
        Q = mx.as_strided(Q, (mha_shape)).transpose([0, 2, 1, 3])
        V = mx.as_strided(V, (mha_shape)).transpose([0, 2, 1, 3])
        attn_weights = (Q @ K.transpose([0, 1, 3, 2])) / math.sqrt(Q.shape[-1])
        attn_weights = attn_weights + self._causal_mask[:T, :T]
        attn_weights = mx.softmax(attn_weights, axis=-1)
        attn_weights = self.attn_dropout(attn_weights)
        o = (attn_weights @ V)
        o = o.transpose([0, 2, 1, 3]).reshape((B, T, head_size))
        o = self.c_proj(self.resid_dropout(o))
        return o

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(n_emb, 4 * n_emb)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)
    def __call__(self, x):
        x = self.gelu(self.c_fc(x))
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()
        self.mha = MultiHeadAttention()
        self.ln_1 = nn.LayerNorm(dims=n_emb)
        self.ln_2 = nn.LayerNorm(dims=n_emb)
    def __call__(self, x):
        x = x + self.mha(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.wte = nn.Embedding(vocab_size, n_emb)
        self.wpe = nn.Embedding(ctx_len, n_emb)
        self.blocks = nn.Sequential(
            *[Block() for _ in range(n_layers)],
        )
        self.ln_f = nn.LayerNorm(dims=n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)
        self._init_parameters()
    
    def __call__(self, x):
        B, T = x.shape
        tok_emb = self.wte(x)
        pos_emb = self.wpe(mx.arange(T))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
    
    def generate(self, max_new_tokens):
        ctx = mx.zeros((1, 1), dtype=mx.int32)
        for _ in range(max_new_tokens):
          logits = self(ctx[:, -ctx_len:])
          logits = logits[:, -1, :]
          next_tok = mx.random.categorical(logits, num_samples=1)
          ctx = mx.concatenate((ctx, next_tok), axis=1)
        return ctx
    
    def _init_parameters(self):
        normal_init = nn.init.normal(mean=0.0, std=0.02)
        residual_init = nn.init.normal(mean=0.0, std=(0.02 / math.sqrt(2 * n_layers)))
        new_params = []
        for name, module in self.named_modules():
            if isinstance(module, nn.layers.linear.Linear):
                if 'c_proj' in name:
                    new_params.append((name + '.weight', residual_init(module.weight)))
                else:
                    new_params.append((name + '.weight', normal_init(module.weight)))
                if 'bias' in module:
                    new_params.append((name + '.bias', mx.zeros(module.bias.shape)))
            elif isinstance(module, nn.layers.embedding.Embedding):
                new_params.append((name + '.weight', normal_init(module.weight)))
        self = self.update(utils.tree_unflatten(new_params)) 