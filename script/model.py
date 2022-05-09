import torch
from torch import nn
import torch.nn.functional as F

from modules.transformer import TransformerEncoder


class MULTModel(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a MulT model.
        l:box   (bn,14,4)
        a:img   (bn,14,2048)
        v:label (bn,14,6)
        """
        super(MULTModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = 2048, 4, 6
        self.d_l, self.d_a, self.d_v = 30, 30, 30
        self.vonly = True
        self.aonly = True
        self.lonly = True
        self.num_heads = hyp_params.n_heads
        self.layers = 5             # 5
        self.attn_dropout = 0.1     # 0.1
        self.attn_dropout_a = 0.0   # 0.0
        self.attn_dropout_v = 0.0   # 0.0
        self.relu_dropout = 0.1     # 0.1
        self.res_dropout = 0.1      # 0.1
        self.out_dropout = 0.0      # 0.0
        self.embed_dropout = 0.25   # 0.25
        self.attn_mask = True       # True

        combined_dim = self.d_l + self.d_a + self.d_v

        self.partial_mode = self.lonly + self.aonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = 2 * self.d_l   # assuming d_l == d_a == d_v
        else:
            combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
        
        output_dim = 6

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout   # 30,   0.1
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a # 30,   0.0
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v # 30,   0.0
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout # 60,   0.1
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout # 60,   0.1
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout # 60,   0.1
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,   #0.1
                                  res_dropout=self.res_dropout,     # 0.1
                                  embed_dropout=self.embed_dropout, # 0.25
                                  attn_mask=self.attn_mask)         # 0.1
            
    def forward(self, x_l, x_a, x_v):
        '''
        l:img   (bn,14,2048)
        a:box   (bn,14,4)
        v:label (bn,14,6)
        '''
        # [3,50,300]
        # [3,375,5]
        # [3,500,20]
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training) # [3,300,50]
        x_a = x_a.transpose(1, 2)   # [3,5,375]
        x_v = x_v.transpose(1, 2)   # [3,20,500]
       
        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)   # [3,30,50]
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)   # [3,30,375]
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)   # [3,30,500]
        proj_x_a = proj_x_a.permute(2, 0, 1)    # [50,3,30]
        proj_x_v = proj_x_v.permute(2, 0, 1)    # [375,3,30]
        proj_x_l = proj_x_l.permute(2, 0, 1)    # [500,3,30]

        if self.lonly:
            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls = self.trans_l_mem(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction

        if self.aonly:
            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l) # l as value passed [18,2,30]
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v) # v as value passed [18,2,30]
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2) # [18,2,60]`
            h_as = self.trans_a_mem(h_as)   # l&v atten self
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l) # pass l
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a) # pass a
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]
        
        if self.partial_mode == 3:
            last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        return output, last_hs