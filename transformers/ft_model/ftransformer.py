''' Module containing class definitions for a transformer implemented to 
    manipulate sequences of floats
''' 

import torch
import torch.nn as nn
import numpy as np
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_p, max_len):
        super().__init__()

        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, d_model)
        positions_list = torch.arange(0, max_len, dtype=torch.long).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, d_model, 2).long() * (-math.log(10000.0)) / d_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/d_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/d_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor, train=False) -> torch.tensor:
        # Residual connection + pos encoding
        
        # If training, apply dropout
        if train:
            return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])
            
        # If not, return without dropout
        else:
            return token_embedding + self.pos_encoding[:token_embedding.size(0), :]
            
class FloatTransformer(nn.Module):
    def __init__(self, d_model, n_head, n_layers, device, in_element_dim = 1, out_element_dim = 1, pos_encoding = False):
        ''' Initialize transformer model. Right now, dropout probability is set to 0 as that is what is found to work
        with float manipulation
        
            Args:
                d_model (int): dimension of embedding
                n_head (int): number of attention heads
                n_layers (int): number of encoders/decoders in encoder and decoder blocks
                in_element_dim (int): The dimensionality of each element of a sequence inputted to the transformer. Default
                set to 1
                out_element_dim (int): The dimensionality of each element of a sequence outputted by the transformer. Default 
                set to 1
        '''
        super().__init__()
        
        ## Define dimensionality of each object of transformer
        # N: Batch num
        # S: Sequence length
        # T: Target length
        
        # Initialize parameters + objects
        self.d_model = d_model
        self.device = device
        
        # Input dimensions: N x S x in_element_dim ---> Output dimensions: N x S x d_model
        self.embedding = nn.Linear(in_element_dim, d_model)
        
        # If doing positional encoding, do so. If not, leave as identity
        if pos_encoding:
        
            # Input dimensions: N x S x d_model ---> Output dimensions: N x S x d_model
            self.positional_encoder = PositionalEncoding(
                d_model = d_model, 
                dropout_p = 0.0,
                max_len = 100)
                
        else:
        
            # Input dimensions: N x S x d_model ---> Output dimensions: N x S x d_model
            self.positional_encoder = nn.Identity()

        # Input dimensions: src: N x S x d_model & tgt: N x T x d_model ---> Output dimensions: N x T x d_model
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dropout = 0 )
        
        # Input dimensions: N x T x d_model ---> Output dimensions: N x T x out_element_dim
        self.out = nn.Linear(d_model, out_element_dim) 
        
    def get_tgt_mask(self, size) -> torch.Tensor:
        ''' Generate square tensor of length size, where the lower triangular entries of the matrix
        are set to 0 (true) and the rest are set to -inf (false)
        
            Args:
                size (int): size of square tensor
                
            Out:
                mask
        '''
        
        # Create lower triangular matrix where triangular entries are 1 and the rest are 0
        mask = torch.tril(torch.ones(size, size) == 1)
        mask = mask.float()
        
        # Turn zeros to -inf
        mask = mask.masked_fill(mask == 0, float('-inf'))
       
        # Turn ones in the matrix to zeros
        mask = mask.masked_fill(mask == 1, float(0.0))
        
        return mask
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, 
                tgt_key_padding_mask=None, verbose=False, train=False):
        ''' Forward method of transformer. Let the following symbols be defined as below:
            N - batch num
            S - src sequence length
            A - src element size
            T - tgt sequence length
            B - tgt element size
            Important: src and tgt tensors must have three dimensions in order to use the forward
            method properly.
            
            Args:
                src (tensor): input sequence(s)   N x S x A
                tgt (tensor): target sequence(s)  N x T x B
                src_mask (tensor): mask for src input  S x S
                tgt_mask (tensor): mask for tgt input  T x T
                src_key_padding_mask (tensor): mask for any padding after EOS  N x S
                tgt_key_padding_mask (tensor): mask for any padding after EOS  N x L
                verbose (bool): boolean for printing out intermediate values
                train (bool): boolean for whether or not to apply dropout w/ positional encoding
        '''
        
        # Apply embeddings
        src_emb = (self.embedding(src) * np.sqrt(self.d_model)).to(self.device)
        tgt_emb = (self.embedding(tgt) * np.sqrt(self.d_model)).to(self.device)
        
        src_pemb = self.positional_encoder(src_emb)
        tgt_pemb = self.positional_encoder(tgt_emb)
        
        # Do so that length is first?
        src_pemb = src_pemb.permute(1, 0, 2)
        tgt_pemb = tgt_pemb.permute(1, 0, 2)
        
        transformer_output = self.transformer(src_pemb, tgt_pemb, src_mask=src_mask, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask).to(self.device)
        
        output = self.out(transformer_output).to(self.device)
    
        # Repermute so that batch size N is first again :>
        output = output.permute(1, 0, 2).to(self.device)
        
        if verbose:
            print(f'\nTracing through the Transformer\n')
            print(f'Input: {src}')
            print(f'Target: {tgt}')
            
            print(f'\nAfter Embeddings')
            print(f'Embedded input: {src_emb}')
            print(f'Embedded target: {tgt_emb}')
            
            print(f'\nAfter Positional Encoding')
            print(f'Positionally encoded input: {src_pemb}')
            print(f'Positionally encoded target: {tgt_pemb}')
            
            print(f'\nAfter Encoder + Decoder Blocks')
            print(f'Output from transformer: {transformer_output}')
            
            print(f'Final output: {output}')
        
        return output