import torch
from torch import nn

class MultiheadAttentionSIM(nn.Module):
    """     
    Omits the last step to multiply the attention weights with the values to produce att weights. 
    > we are only interested in the similarites.
    """ 
    def __init__(self, d_model, num_heads, seq_len, lin_out=False):
        super(MultiheadAttentionSIM, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.seq_len = seq_len
        self.lin_out = lin_out

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)

        # below can be used to downscale the the output of the concataned 8 heights to a [topK, topK] map that we can use. Like maybe
        if lin_out:
            self.linear_out = nn.Linear(self.num_heads * seq_len * seq_len, seq_len * seq_len)
        else:
            self.linear_out = nn.Linear(self.num_heads * seq_len, seq_len)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, mask=None):
        batch_size, seq_len, _ = query.size()

        query = self.split_heads(self.W_q(query), batch_size)
        key = self.split_heads(self.W_k(key), batch_size)

        query /= self.head_dim**0.5

        scores = torch.matmul(query, key.transpose(-2, -1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        # # print("attn_weights after softmax", attn_weights.size())
        
        # Concatenate attention weights along the head dimension
        attn_weights = attn_weights.view(batch_size, self.num_heads, seq_len, seq_len)
        if not self.lin_out:
            attn_weights = attn_weights.mean(dim=1)  # Average across heads. We could use some kind of linear to map it down. 
        else:
            # Linear mapping to output with shape [batch_size, seq_len, seq_len]
            # attn_weights = self.linear_out(attn_weights)
            attn_weights = attn_weights.view(batch_size, -1)
            # print("attn_weights view()", attn_weights.size())
            attn_weights = self.linear_out(attn_weights)
            # print("attn_weights linear()", attn_weights.size())
            attn_weights = attn_weights.view(batch_size, seq_len, seq_len)
            # print("attn_weights reshaped()", attn_weights.size())
        # print("attn_weights", attn_weights.size())

        return attn_weights