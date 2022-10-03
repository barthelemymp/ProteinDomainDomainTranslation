""" 
The code bellow was taken from: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py
and modified to fit our needs.

"""

import copy
from typing import Optional, Any
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
import os
import subprocess
import pandas as pd
import math
import tempfile


class ProteinTransformer(nn.Module):
    r"""A transformer model. User is able to modify the attributes as needed. The architecture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
    model with corresponding parameters.

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::
        >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)

    Note: A full example to apply nn.Transformer module for the word language model is available in
    https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ProteinTransformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first,
                                                    **factory_kwargs)
            encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first,
                                                    **factory_kwargs)
            decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.batch_first = batch_first

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).

        Shape:
            - src: :math:`(S, N, E)`, `(N, S, E)` if batch_first.
            - tgt: :math:`(T, N, E)`, `(N, T, E)` if batch_first.
            - src_mask: :math:`(S, S)`.
            - tgt_mask: :math:`(T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(N, S)`.

            Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.
            [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a ByteTensor is provided, the non-zero positions will be ignored while the zero
            positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

            - output: :math:`(T, N, E)`, `(N, T, E)` if batch_first.

            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.

            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number

        Examples:
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """

        if not self.batch_first and src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output


    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)



class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output



class TransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False,
                 device=None, dtype=None, saveAttention=False) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.saveAttention = saveAttention
        self.savedAttention = None

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src



class TransformerDecoderLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)

    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """
    __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False, device=None, dtype=None, saveAttention=False) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        
        self.saveAttention = saveAttention
        self.savedAttention = None

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]

        
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if self.saveAttention==True:
            tgt2, self.savedAttention = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        else:
            tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                       key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt



def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
    
    
    

    
    

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    source: https://github.com/pytorch/tutorials/blob/011ae8a6d47a960935d0401acda71d0e400088d6/advanced_source/ddp_pipeline.py#L43

    """

    def __init__(self, d_model, dropout=0.1, max_len=25,device='cpu'):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)[:,:pe[:, 0::2].shape[1]]
        pe[:, 1::2] = torch.cos(position * div_term)[:,:pe[:, 1::2].shape[1]]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        print(self.pe.device)
        print(self.pe.device)
        
    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        
        x = (x + self.pe[:x.size(0), :]).to(self.device)
        return self.dropout(x)
    
    
    


class StructuralAlignedEncoder(nn.Module):
    def __init__(self, d_model, fastapath, pdbPath, chain, dropout=0.1, max_len=25,device='cpu' ):
        super(StructuralAlignedEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.xstop = d_model//4
        self.ystop = 2*(d_model//4)
        self.zstop = 3*(d_model//4)
        prefix="csvTemp"
        tempFile=next(tempfile._get_candidate_names())
        subprocess.run(["julia", "mapHmmToPDB.jl", fastapath, pdbPath, chain, tempFile])
        df = pd.read_csv(tempFile, header=None)
        xs = torch.tensor(df[:][0]).unsqueeze(1)
        ys = torch.tensor(df[:][1]).unsqueeze(1)
        zs = torch.tensor(df[:][2]).unsqueeze(1)
        pe = torch.zeros(max_len, d_model).to(device)
        pex = torch.zeros(max_len, self.xstop)
        pey = torch.zeros(max_len, self.xstop)
        pez = torch.zeros(max_len, self.xstop)
        pel = torch.zeros(max_len, d_model-self.zstop+1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_termCoordinate = torch.exp(torch.arange(0, self.xstop, 2).float() * (-math.log(10000.0) / d_model))
        div_termPos = torch.exp(torch.arange(0, d_model-self.zstop, 2).float() * (-math.log(10000.0) / d_model))
        pex[:, 0::2] = torch.sin(xs * div_termCoordinate)
        pex[:, 1::2] = torch.cos(xs * div_termCoordinate)
        pey[:, 0::2] = torch.sin(ys * div_termCoordinate)
        pey[:, 1::2] = torch.cos(ys * div_termCoordinate)
        pez[:, 0::2] = torch.sin(zs * div_termCoordinate)
        pez[:, 1::2] = torch.cos(zs * div_termCoordinate)
        pel[:, 0::2] = torch.sin(position * div_termPos)
        pel[:, 1::2] = torch.cos(position * div_termPos)
        pe[:,0:self.xstop] = pex
        pe[:,self.xstop:self.ystop] = pey
        pe[:,self.ystop:self.zstop] = pez
        pe[:,self.zstop:] = pel[:,:-1]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        print(self.pe.device)
        os.remove(tempFile)
    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """   
        x = (x + self.pe[:x.size(0), :]).to(self.device)
        return self.dropout(x)
    
    
    
class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        src_posEnc,
        trg_posEnc,
        device,
        onehot=True,
        sparseEmbed = False,
    ):
        super(Transformer, self).__init__()
        self.device = device
        self.src_position_embedding = src_posEnc
        self.trg_position_embedding = trg_posEnc
        self.embedding_size = embedding_size
        self.onehot = onehot
        if onehot==False:
            self.embed_tokens = nn.Embedding(src_vocab_size, embedding_size, sparse=sparseEmbed)
        self.transformer = ProteinTransformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.trg_vocab_size =trg_vocab_size
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx
        self.sparseEmbed = sparseEmbed

        
        
    
    def OneHot(self,in_tensor):
        
        seq_length, N = in_tensor.shape
        out_one_hot = torch.zeros((in_tensor.shape[0], in_tensor.shape[1],self.embedding_size))
        for i in range(seq_length):
            for j in range(N):
                c = in_tensor[i,j]
                out_one_hot[i,j,c] = 1
        return out_one_hot
    
    def make_src_mask(self, src):
        """
        If we have padded the source input (to be of the same size among the same batch I guess)
        there is no need to do computation for them, so this function masks the 
        padded parts.
        src is sequence to the encoder 
        """
        padPos = self.src_pad_idx

        src_mask = src[:,:].transpose(0, 1) == padPos

        return src_mask.to(self.device)

    def forward(self, src, trg):

        src_seq_length = src.shape[0]
        trg_seq_length= trg.shape[0]

        src_padding_mask = self.make_src_mask(src)

        if self.onehot==False:
            if len(src.shape)==2:
                src = self.embed_tokens(src)
            else:
                if self.sparseEmbed:
                    src = torch.sparse.mm(src, self.embed_tokens.weight)
                else:
                    src = torch.matmul(src, self.embed_tokens.weight)
            if len(trg.shape)==2:
                trg = self.embed_tokens(trg)
            else:
                if self.sparseEmbed:
                    print(trg.shape, self.embed_tokens.weight.shape)
                    trg = torch.sparse.mm(trg, self.embed_tokens.weight)
                else:
                    trg = torch.matmul(trg, self.embed_tokens.weight)
                
            
        embed_src = self.src_position_embedding.forward(src)
        embed_trg = self.trg_position_embedding.forward(trg)


        
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out
    
    def sample(self, inp, max_len, nsample=1, method="simple"):
        """ sample output protein given input proteins:
                -nsample only relevant if inp consist of one sample.
                -method = simple means that the output is sampled using conditional distrib but we can not backpropagate trough the samples
                -method = gumbel: the sample are backpropagable.
            return samples sequence in the onehot format in very case"""
        if self.onehot:
            sos = inp[0,0,:]
            eos = inp[-1,0,:]
        else:
            sos = torch.nn.functional.one_hot(inp[0,0], num_classes=self.trg_vocab_size)
            eos = torch.nn.functional.one_hot(inp[-1,0], num_classes=self.trg_vocab_size)
        if inp.shape[1]!=1:
            nsample=inp.shape[1]
            inp_repeted = inp
        else:
            if self.onehot:
                inp_repeted = inp[:,0,:].unsqueeze(1).repeat(1, nsample, 1)
            else:
                inp_repeted = inp[:,0].unsqueeze(1).repeat(1, nsample)
                
        
        if method=="simple":
            outputs = torch.zeros(max_len, nsample, self.trg_vocab_size).to(self.device)
            outputs[0,:,:] = sos.unsqueeze(0).repeat(nsample, 1)
            for i in range(1,max_len):
                output = self.forward(inp_repeted, outputs[:i])
                logits = output.reshape(-1,self.trg_vocab_size)
                best_guess = torch.distributions.Categorical(logits=logits).sample()
                best_guess = torch.nn.functional.one_hot(best_guess, num_classes=self.trg_vocab_size).reshape(-1,nsample,self.trg_vocab_size)
                outputs[i,:,:]= best_guess[-1,:,:]

            outputs[-1,:,:] = eos.unsqueeze(0).repeat(nsample, 1)

            return outputs

        if method=="gumbel":
            outputs = torch.zeros(max_len, nsample, self.trg_vocab_size).to(self.device)
            outputs[0,:,:] = sos.unsqueeze(0).repeat(nsample, 1)
            for i in range(1,max_len):
                output = self.forward(inp_repeted, outputs[:i])
                best_guess = torch.nn.functional.gumbel_softmax(output, hard=True, dim=2)
                outputs[i,:,:]= best_guess[-1,:,:]

            outputs[-1,:,:] = eos.unsqueeze(0).repeat(nsample, 1)
            return outputs
            
        if method=="bestguess":
            outputs = torch.zeros(max_len, nsample, self.trg_vocab_size).to(self.device)
            outputs[0,:,:] = sos.unsqueeze(0).repeat(nsample, 1)
            for i in range(1,max_len):
                with torch.no_grad():
                    output = self.forward(inp_repeted, outputs[:i,:,:])
                best_guess = output.argmax(2)[-1, :].item()
                outputs[i,:,:]= torch.nn.functional.one_hot(best_guess, num_classes=self.trg_vocab_size)

            outputs[-1,:,:] = eos.unsqueeze(0).repeat(nsample, 1)
            return outputs
        
    def pseudosample(self, inp, target, nsample=1, method="simple"):
        if self.onehot:
            sos = inp[0,0,:]
            eos = inp[-1,0,:]
        else:
            sos = torch.nn.functional.one_hot(inp[0,0], num_classes=self.trg_vocab_size)
            eos = torch.nn.functional.one_hot(inp[-1,0], num_classes=self.trg_vocab_size)
        if inp.shape[1]!=1:
            nsample=inp.shape[1]
        else:
            if self.onehot:
                inp_repeted = inp[:,0,:].unsqueeze(1).repeat(1, nsample, 1)
            else:
                inp_repeted = inp[:,0].unsqueeze(1).repeat(1, nsample)
                
            
        

        if method=="simple":
            if self.onehot:
                outputs = torch.zeros(target.shape[0], nsample, target.shape[2]).to(self.device)
                outputs[0,:,:] = sos.unsqueeze(0).repeat(nsample, 1)
                output = self.forward(inp, target[:-1, :])
                prob = torch.nn.functional.softmax(output.clone().detach(),dim=2).reshape(-1,inp.shape[2])
                best_guess = torch.multinomial(prob, nsample, replacement=True)
                best_guess = torch.nn.functional.one_hot(best_guess, num_classes=self.trg_vocab_size).reshape(-1,nsample,self.trg_vocab_size)
                outputs[1:,:,:]= best_guess
                outputs[-1,:,:] = eos.unsqueeze(0).repeat(nsample, 1)
            else:
                print("todo")
            return outputs

                
            
        if method=="gumbel":
            if self.onehot:
                outputs = torch.zeros(target.shape[0], nsample, target.shape[2]).to(self.device)
                outputs[0,:,:] = sos.unsqueeze(0).repeat(nsample, 1)
    
                output = self.forward(inp, target[:-1, :])
                best_guess = torch.nn.functional.gumbel_softmax(output, hard=True, dim=2)
                outputs[1:,:,:]= best_guess
                outputs[-1,:,:] = eos.unsqueeze(0).repeat(nsample, 1)
            else:
                outputs = torch.zeros(target.shape[0], nsample, self.trg_vocab_size).to(self.device)
                outputs[0,:,:] = sos.unsqueeze(0).repeat(nsample, 1)
                output = self.forward(inp, target[:-1, :])
                best_guess = torch.nn.functional.gumbel_softmax(output, hard=True, dim=2)
                print(best_guess)
                outputs[1:,:,:] = best_guess
                outputs[-1,:,:] = eos.unsqueeze(0).repeat(nsample, 1)

            return outputs
            
        if method=="bestguess":
            outputs = torch.zeros(target.shape[0], 1, target.shape[2]).to(self.device)
            outputs[0,:,:] = sos.unsqueeze(0).repeat(1, 1)

            with torch.no_grad():
                output = self.forward(inp_repeted, target[:-1, :])
            best_guess = output.argmax(2)[-1, :].item()
            outputs[1:,:,:]= torch.nn.functional.one_hot(best_guess, num_classes=self.trg_vocab_size)

            outputs[-1,:,:] = eos.unsqueeze(0).repeat(nsample, 1)
            return outputs
        