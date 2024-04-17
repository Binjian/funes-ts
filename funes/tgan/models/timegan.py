import torch
import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd
import math
import torch.nn as nn

import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder,TransformerEncoderLayer

__all__ = [
    "TimeGAN",
    "UnknowGANType",
    "compute_gradient_penalty",
    "EmbeddingNetwork",
    "GeneratorNetwork",
    "DiscriminatorNetwork",
    "RecoveryNetwork",
    "SupervisorNetwork",
]

lambda_gp = 10
Tensor = torch.cuda.FloatTensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):

        # x = x + self.pe[:x.size(0), :]
        x = x + self.pe[:,:x.size(1), :]

        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.1,batch_first=True,norm_first=True):
        super(TransformerModel, self).__init__()

        self.src_mask = None
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.pos_encoder = PositionalEncoding(nhid, dropout,max_len=8000)
        # self.pos_encoder = nn.Parameter(torch.zeros(1,50, nhid))  # tts gan position encoder
        encoder_layers = TransformerEncoderLayer(nhid, nhead, nhid, dropout, batch_first=self.batch_first,norm_first=self.norm_first)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.linear = nn.Linear(ninp, nhid)
        self.ninp = ninp

    def forward(self,X):
        X = self.linear(X)*math.sqrt(self.ninp)
        X = self.pos_encoder(X)

        # X = self.linear(X) + self.pos_encoder # tts gan pos encoder

        out = self.transformer_encoder(X)
        return out

class UnknowGANType(Exception):
    pass

class LayerNormGRUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormGRUCell, self).__init__()

        self.ln_i2h = torch.nn.LayerNorm(2*hidden_size, elementwise_affine=False)
        self.ln_h2h = torch.nn.LayerNorm(2*hidden_size, elementwise_affine=False)
        self.ln_cell_1 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln_cell_2 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.i2h = torch.nn.Linear(input_size, 2 * hidden_size, bias=bias)
        self.h2h = torch.nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        self.h_hat_W = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.h_hat_U = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.hidden_size = hidden_size
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, h):

        h = h
        h = h.view(h.size(0), -1)
        x = x.view(x.size(0), -1)

        # Linear mappings
        i2h = self.i2h(x)
        h2h = self.h2h(h)

        # Layer norm
        i2h = self.ln_i2h(i2h)
        h2h = self.ln_h2h(h2h)

        preact = i2h + h2h

        # activations
        gates = preact[:, :].sigmoid()
        z_t = gates[:, :self.hidden_size]
        r_t = gates[:, -self.hidden_size:]

        # h_hat
        h_hat_first_half = self.h_hat_W(x)
        h_hat_last_half = self.h_hat_U(h)

        # layer norm
        h_hat_first_half = self.ln_cell_1( h_hat_first_half )
        h_hat_last_half = self.ln_cell_2( h_hat_last_half )

        h_hat = torch.tanh(  h_hat_first_half + torch.mul(r_t,   h_hat_last_half ) )

        h_t = torch.mul( 1-z_t , h ) + torch.mul( z_t, h_hat)

        # Reshape for compatibility

        h_t = h_t.view( h_t.size(0), -1)
        return h_t


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True
    )
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


class EmbeddingNetwork(torch.nn.Module):
    """The embedding network (encoder) for TimeGAN"""

    def __init__(self, args):
        super(EmbeddingNetwork, self).__init__()
        self.feature_dim = args.feature_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len
        self.encode_mode = args.encode_mode
        self.batch_size = args.batch_size
        self.heads = args.heads

        if self.encode_mode == 'layer_norm_gru':
            self.emb_rnn = torch.nn.Sequential()
            for i in range(self.num_layers):
                if i == 0:
                    self.emb_rnn.add_module('emb_layer_norm_gru_cell' + str(i), LayerNormGRUCell(self.feature_dim, self.hidden_dim))
                else:
                    self.emb_rnn.add_module('emb_layer_norm_gru_cell' + str(i), LayerNormGRUCell(self.hidden_dim, self.hidden_dim))
        elif self.encode_mode == 'gru':
            # Embedder Architecture
            self.emb_rnn = torch.nn.GRU(
                input_size=self.feature_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
            )
        elif self.encode_mode == 'transformer':
            self.emb_rnn = TransformerModel(ninp=self.feature_dim,nhead=self.heads,nhid=self.hidden_dim,nlayers=self.num_layers,
                                            dropout=0,batch_first=True,norm_first=True)

        # self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.emb_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.emb_sigmoid = torch.nn.Sigmoid()

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference:
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.emb_rnn.named_parameters():
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "bias_ih" in name:
                    param.data.fill_(1)
                elif "bias_hh" in name:
                    param.data.fill_(0)
            for name, param in self.emb_linear.named_parameters():
                if "weight" in name:
                    torch.nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    param.data.fill_(0)

    def forward(self, X, T):
        """Forward pass for embedding features from original space into latent space
        Args:
            - X: input time-series features (B x S x F)
            - T: input temporal information (B)
        Returns:
            - H: latent space embeddings (B x S x H)
        """

        if self.encode_mode == 'layer_norm_gru':
            h_init = torch.nn.init.xavier_uniform_(torch.empty(X.shape[0],self.hidden_dim)).to('cuda')
            # h_init = torch.zeros([self.batch_size,self.hidden_dim],device='cuda')  # batch size,hidden dim
            H_o = []
            for t in range(T[0]):
                h_layers = []
                if t == 0:
                    for idx in range(len(self.emb_rnn)):  # layer: idx
                        if idx == 0:
                            h = self.emb_rnn[idx](X[:, t, :], h_init)
                        else:
                            h = self.emb_rnn[idx](h, h_init)
                        h_layers.append(h)
                else:
                    for idx in range(len(self.emb_rnn)):
                        if idx == 0:
                            h = self.emb_rnn[idx](X[:, t, :], h_tmp[idx])
                        else:
                            h = self.emb_rnn[idx](h, h_tmp[idx])
                        h_layers.append(h)
                h_tmp = h_layers
                H_o.append(h_layers[-1])
            H_o = torch.stack(H_o).permute([1, 0, 2])
        elif self.encode_mode == 'gru':
            # Dynamic RNN input for ignoring paddings
            X_packed = torch.nn.utils.rnn.pack_padded_sequence(
                input=X, lengths=T, batch_first=True, enforce_sorted=False
            )

            # 128 x 100 x 71
            H_o, H_t = self.emb_rnn(X_packed)

            # Pad RNN output back to sequence length
            H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
                sequence=H_o,
                batch_first=True,
                padding_value=self.padding_value,
                total_length=self.max_seq_len,
            )

        elif self.encode_mode == 'transformer':
            H_o = self.emb_rnn(X)
            # H_o = self.layer_norm(H_o)
        # 128 x 100 x 10
        logits = self.emb_linear(H_o)

        # 128 x 100 x 10
        H = self.emb_sigmoid(logits)

        return H


class RecoveryNetwork(torch.nn.Module):
    """The recovery network (decoder) for TimeGAN"""

    def __init__(self, args):
        super(RecoveryNetwork, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.feature_dim = args.feature_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len
        self.encode_mode = args.encode_mode
        self.batch_size = args.batch_size
        self.heads = args.heads

        if self.encode_mode == 'layer_norm_gru':
            self.rec_rnn = torch.nn.Sequential()
            for i in range(self.num_layers):
                if i == 0:
                    self.rec_rnn.add_module('rec_layer_norm_gru_cell' + str(i),
                                            LayerNormGRUCell(self.hidden_dim, self.hidden_dim))
                else:
                    self.rec_rnn.add_module('rec_layer_norm_gru_cell' + str(i),
                                            LayerNormGRUCell(self.hidden_dim, self.hidden_dim))
        elif self.encode_mode == 'gru':
            # Recovery Architecture
            self.rec_rnn = torch.nn.GRU(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
            )
        elif self.encode_mode == 'transformer':
            self.rec_rnn = TransformerModel(ninp=self.hidden_dim, nhead=self.heads, nhid=self.hidden_dim,
                                            nlayers=self.num_layers, dropout=0, batch_first=True,norm_first=True)
        # self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.rec_linear = torch.nn.Linear(self.hidden_dim, self.feature_dim)
        self.rec_sigmoid = torch.nn.Sigmoid()

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference:
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.rec_rnn.named_parameters():
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "bias_ih" in name:
                    param.data.fill_(1)
                elif "bias_hh" in name:
                    param.data.fill_(0)
            for name, param in self.rec_linear.named_parameters():
                if "weight" in name:
                    torch.nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    param.data.fill_(0)

    def forward(self, H, T):
        """Forward pass for the recovering features from latent space to original space
        Args:
            - H: latent representation (B x S x E)
            - T: input temporal information (B)
        Returns:
            - X_tilde: recovered data (B x S x F)
        """

        if self.encode_mode == 'layer_norm_gru':
            h_init = torch.nn.init.xavier_uniform_(torch.empty(H.shape[0],self.hidden_dim)).to('cuda')
            H_o = []
            for t in range(T[0]):
                h_layers = []
                if t == 0:
                    for idx in range(len(self.rec_rnn)):  # layer: idx
                        if idx == 0:
                            h = self.rec_rnn[idx](H[:, t, :], h_init)
                        else:
                            h = self.rec_rnn[idx](h, h_init)
                        h_layers.append(h)
                else:
                    for idx in range(len(self.rec_rnn)):
                        if idx == 0:
                            h = self.rec_rnn[idx](H[:, t, :], h_tmp[idx])
                        else:
                            h = self.rec_rnn[idx](h, h_tmp[idx])
                        h_layers.append(h)
                h_tmp = h_layers
                H_o.append(h_layers[-1])
            H_o = torch.stack(H_o).permute([1, 0, 2])
        elif self.encode_mode == 'gru':
            # Dynamic RNN input for ignoring paddings
            H_packed = torch.nn.utils.rnn.pack_padded_sequence(
                input=H, lengths=T, batch_first=True, enforce_sorted=False
            )

            # 128 x 100 x 10
            H_o, H_t = self.rec_rnn(H_packed)

            # Pad RNN output back to sequence length
            H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
                sequence=H_o,
                batch_first=True,
                padding_value=self.padding_value,
                total_length=self.max_seq_len,
            )
        elif self.encode_mode == 'transformer':
            H_o = self.rec_rnn(H)
            # H_o = self.layer_norm(H_o)

        # 128 x 100 x 71
        # X_tilde = self.rec_linear(H_o)
        X_tilde = self.rec_sigmoid(self.rec_linear(H_o))

        return X_tilde


class SupervisorNetwork(torch.nn.Module):
    """The Supervisor network (decoder) for TimeGAN"""

    def __init__(self, args):
        super(SupervisorNetwork, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len
        self.encode_mode = args.encode_mode
        self.batch_size = args.batch_size
        self.heads = args.heads

        if self.encode_mode == 'layer_norm_gru':
            self.sup_rnn = torch.nn.Sequential()
            for i in range(self.num_layers):
                if i == 0:
                    self.sup_rnn.add_module('sup_layer_norm_gru_cell' + str(i),
                                            LayerNormGRUCell(self.hidden_dim, self.hidden_dim))
                else:
                    self.sup_rnn.add_module('sup_layer_norm_gru_cell' + str(i),
                                            LayerNormGRUCell(self.hidden_dim, self.hidden_dim))
        elif self.encode_mode == 'gru':
            # Supervisor Architecture
            self.sup_rnn = torch.nn.GRU(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers - 1,
                batch_first=True,
            )
        elif self.encode_mode == 'transformer':
            self.sup_rnn = TransformerModel(ninp=self.hidden_dim, nhead=self.heads, nhid=self.hidden_dim,
                                            nlayers=self.num_layers, dropout=0, batch_first=True,norm_first=True)

        # self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.sup_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.sup_sigmoid = torch.nn.Sigmoid()

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference:
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.sup_rnn.named_parameters():
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "bias_ih" in name:
                    param.data.fill_(1)
                elif "bias_hh" in name:
                    param.data.fill_(0)
            for name, param in self.sup_linear.named_parameters():
                if "weight" in name:
                    torch.nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    param.data.fill_(0)

    def forward(self, H, T):
        """Forward pass for the supervisor for predicting next step
        Args:
            - H: latent representation (B x S x E)
            - T: input temporal information (B)
        Returns:
            - H_hat: predicted next step data (B x S x E)
        """

        if self.encode_mode == 'layer_norm_gru':
            h_init = torch.nn.init.xavier_uniform_(torch.empty(H.shape[0],self.hidden_dim)).to('cuda')
            H_o = []
            for t in range(T[0]):
                h_layers = []
                if t == 0:
                    for idx in range(len(self.sup_rnn)):  # layer: idx
                        if idx == 0:
                            h = self.sup_rnn[idx](H[:, t, :], h_init)
                        else:
                            h = self.sup_rnn[idx](h, h_init)
                        h_layers.append(h)
                else:
                    for idx in range(len(self.sup_rnn)):
                        if idx == 0:
                            h = self.sup_rnn[idx](H[:, t, :], h_tmp[idx])
                        else:
                            h = self.sup_rnn[idx](h, h_tmp[idx])
                        h_layers.append(h)
                h_tmp = h_layers
                H_o.append(h_layers[-1])
            H_o = torch.stack(H_o).permute([1, 0, 2])
        elif self.encode_mode == 'gru':
            # Dynamic RNN input for ignoring paddings
            H_packed = torch.nn.utils.rnn.pack_padded_sequence(
                input=H, lengths=T, batch_first=True, enforce_sorted=False
            )

            # 128 x 100 x 10
            H_o, H_t = self.sup_rnn(H_packed)

            # Pad RNN output back to sequence length
            H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
                sequence=H_o,
                batch_first=True,
                padding_value=self.padding_value,
                total_length=self.max_seq_len,
            )
        elif self.encode_mode == 'transformer':
            H_o = self.sup_rnn(H)
            # H_o = self.layer_norm(H_o)

        # 128 x 100 x 10
        logits = self.sup_linear(H_o)

        # 128 x 100 x 10
        H_hat = self.sup_sigmoid(logits)


        return H_hat


class GeneratorNetwork(torch.nn.Module):
    """The generator network (encoder) for TimeGAN"""

    def __init__(self, args):
        super(GeneratorNetwork, self).__init__()
        self.Z_dim = args.Z_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len
        self.encode_mode = args.encode_mode
        self.batch_size = args.batch_size
        self.heads = args.heads

        if self.encode_mode == 'layer_norm_gru':
            self.gen_rnn = torch.nn.Sequential()
            for i in range(self.num_layers):
                if i == 0:
                    self.gen_rnn.add_module('gen_layer_norm_gru_cell' + str(i),
                                            LayerNormGRUCell(self.Z_dim, self.hidden_dim))
                else:
                    self.gen_rnn.add_module('gen_layer_norm_gru_cell' + str(i),
                                            LayerNormGRUCell(self.hidden_dim, self.hidden_dim))
        elif self.encode_mode == 'gru':
            # Generator Architecture
            self.gen_rnn = torch.nn.GRU(
                input_size=self.Z_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
            )
        elif self.encode_mode == 'transformer':
            self.gen_rnn = TransformerModel(ninp=self.Z_dim, nhead=self.heads, nhid=self.hidden_dim,
                                            nlayers=self.num_layers, dropout=0, batch_first=True,norm_first=True)

        # self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.gen_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gen_sigmoid = torch.nn.Sigmoid()

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference:
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.gen_rnn.named_parameters():
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "bias_ih" in name:
                    param.data.fill_(1)
                elif "bias_hh" in name:
                    param.data.fill_(0)
            for name, param in self.gen_linear.named_parameters():
                if "weight" in name:
                    torch.nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    param.data.fill_(0)

    def forward(self, Z, T):
        """Takes in random noise (features) and generates synthetic features within the latent space
        Args:
            - Z: input random noise (B x S x Z)
            - T: input temporal information
        Returns:
            - H: embeddings (B x S x E)
        """

        if self.encode_mode == 'layer_norm_gru':
            h_init = torch.nn.init.xavier_uniform_(torch.empty(Z.shape[0],self.hidden_dim)).to('cuda')
            H_o = []
            for t in range(T[0]):
                h_layers = []
                if t == 0:
                    for idx in range(len(self.gen_rnn)):  # layer: idx
                        if idx == 0:
                            h = self.gen_rnn[idx](Z[:, t, :], h_init)
                        else:
                            h = self.gen_rnn[idx](h, h_init)
                        h_layers.append(h)
                else:
                    for idx in range(len(self.gen_rnn)):
                        if idx == 0:
                            h = self.gen_rnn[idx](Z[:, t, :], h_tmp[idx])
                        else:
                            h = self.gen_rnn[idx](h, h_tmp[idx])
                        h_layers.append(h)
                h_tmp = h_layers
                H_o.append(h_layers[-1])
            H_o = torch.stack(H_o).permute([1, 0, 2])
        elif self.encode_mode == 'gru':
            # Dynamic RNN input for ignoring paddings
            Z_packed = torch.nn.utils.rnn.pack_padded_sequence(
                input=Z, lengths=T, batch_first=True, enforce_sorted=False
            )

            # 128 x 100 x 71
            H_o, H_t = self.gen_rnn(Z_packed)

            # Pad RNN output back to sequence length
            H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
                sequence=H_o,
                batch_first=True,
                padding_value=self.padding_value,
                total_length=self.max_seq_len,
            )
        elif self.encode_mode == 'transformer':
            H_o = self.gen_rnn(Z)
            # H_o = self.layer_norm(H_o)

        # 128 x 100 x 10
        logits = self.gen_linear(H_o)

        # B x S
        H = self.gen_sigmoid(logits)

        return H


class DiscriminatorNetwork(torch.nn.Module):
    """The Discriminator network (decoder) for TimeGAN"""

    def __init__(self, args):
        super(DiscriminatorNetwork, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len
        self.encode_mode = args.encode_mode
        self.batch_size = args.batch_size
        self.heads = args.heads

        if self.encode_mode == 'layer_norm_gru':
            self.dis_rnn = torch.nn.Sequential()
            for i in range(self.num_layers):
                if i == 0:
                    self.dis_rnn.add_module('dis_layer_norm_gru_cell' + str(i),
                                            LayerNormGRUCell(self.hidden_dim, self.hidden_dim))
                else:
                    self.dis_rnn.add_module('dis_layer_norm_gru_cell' + str(i),
                                            LayerNormGRUCell(self.hidden_dim, self.hidden_dim))
        elif self.encode_mode == 'gru':
            # Discriminator Architecture
            self.dis_rnn = torch.nn.GRU(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
            )
        elif self.encode_mode == 'transformer':
            self.dis_rnn = TransformerModel(ninp=self.hidden_dim, nhead=self.heads, nhid=self.hidden_dim,
                                            nlayers=self.num_layers, dropout=0, batch_first=True,norm_first=True)

        # self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.dis_linear = torch.nn.Linear(self.hidden_dim, 1)

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference:
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.dis_rnn.named_parameters():
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif "bias_ih" in name:
                    param.data.fill_(1)
                elif "bias_hh" in name:
                    param.data.fill_(0)
            for name, param in self.dis_linear.named_parameters():
                if "weight" in name:
                    torch.nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    param.data.fill_(0)

    def forward(self, H, T):
        """Forward pass for predicting if the data is real or synthetic
        Args:
            - H: latent representation (B x S x E)
            - T: input temporal information
        Returns:
            - logits: predicted logits (B x S x 1)
        """

        if self.encode_mode == 'layer_norm_gru':
            h_init = torch.nn.init.xavier_uniform_(torch.empty(self.batch_size,self.hidden_dim)).to('cuda')
            H_o = []
            for t in range(T[0]):
                h_layers = []
                if t == 0:
                    for idx in range(len(self.dis_rnn)):  # layer: idx
                        if idx == 0:
                            h = self.dis_rnn[idx](H[:, t, :], h_init)
                        else:
                            h = self.dis_rnn[idx](h, h_init)
                        h_layers.append(h)
                else:
                    for idx in range(len(self.dis_rnn)):
                        if idx == 0:
                            h = self.dis_rnn[idx](H[:, t, :], h_tmp[idx])
                        else:
                            h = self.dis_rnn[idx](h, h_tmp[idx])
                        h_layers.append(h)
                h_tmp = h_layers
                H_o.append(h_layers[-1])
            H_o = torch.stack(H_o).permute([1, 0, 2])
        elif self.encode_mode == 'gru':

            # Dynamic RNN input for ignoring paddings
            H_packed = torch.nn.utils.rnn.pack_padded_sequence(
                input=H, lengths=T, batch_first=True, enforce_sorted=False
            )

            # 128 x 100 x 10
            H_o, H_t = self.dis_rnn(H_packed)

            # Pad RNN output back to sequence length
            H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
                sequence=H_o,
                batch_first=True,
                padding_value=self.padding_value,
                total_length=self.max_seq_len,
            )
        elif self.encode_mode == 'transformer':
            H_o = self.dis_rnn(H)
            # H_o = self.layer_norm(H_o)

        logits = self.dis_linear(H_o).squeeze(-1)

        return logits


class TimeGAN(torch.nn.Module):
    """Implementation of TimeGAN (Yoon et al., 2019) using PyTorch
    Reference:
    - https://papers.nips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html
    - https://github.com/jsyoon0823/TimeGAN
    """

    def __init__(self, args):
        """
        :param args:
        :param args.gradient_type:
                        "vanilla",
                        "wasserstein",
                        "wasserstein_gp"
        """
        super(TimeGAN, self).__init__()
        self.device = args.device
        self.feature_dim = args.feature_dim
        self.Z_dim = args.Z_dim
        self.hidden_dim = args.hidden_dim
        self.max_seq_len = args.max_seq_len
        self.batch_size = args.batch_size
        self.gradient_type = args.gradient_type

        self.embedder = EmbeddingNetwork(args)
        self.recovery = RecoveryNetwork(args)
        self.generator = GeneratorNetwork(args)
        self.supervisor = SupervisorNetwork(args)
        self.discriminator = DiscriminatorNetwork(args)

    def _recovery_forward(self, X, T):
        """The embedding network forward pass and the embedder network loss
        Args:
            - X: the original input features
            - T: the temporal information
        Returns:
            - E_loss: the reconstruction loss
            - X_tilde: the reconstructed features
        """
        # Forward Pass
        H = self.embedder(X, T)
        X_tilde = self.recovery(H, T)

        # For Joint training
        H_hat_supervise = self.supervisor(H, T)
        G_loss_S = torch.nn.functional.mse_loss(
            H_hat_supervise[:, :-1, :], H[:, 1:, :]
        )  # Teacher forcing next output

        # Reconstruction Loss
        E_loss_T0 = torch.nn.functional.mse_loss(X_tilde, X)
        E_loss0 = 10 * torch.sqrt(E_loss_T0)
        E_loss = E_loss0 + 0.1 * G_loss_S


        return E_loss, E_loss0, E_loss_T0

    def _supervisor_forward(self, X, T):
        """The supervisor training forward pass
        Args:
            - X: the original feature input
        Returns:
            - S_loss: the supervisor's loss
        """
        # Supervision Forward Pass
        H = self.embedder(X, T)
        H_hat_supervise = self.supervisor(H, T)

        # Supervised loss
        S_loss = torch.nn.functional.mse_loss(
            H_hat_supervise[:, :-1, :], H[:, 1:, :]
        )  # Teacher forcing next output


        return S_loss

    def _discriminator_forward(self, X, T, Z, gamma=1):
        """The discriminator forward pass and adversarial loss
        Args:
            - X: the input features
            - T: the temporal information
            - Z: the input noise
        Returns:
            - D_loss: the adversarial loss
        """
        # Real
        H = self.embedder(X, T).detach()

        # Generator
        E_hat = self.generator(Z, T).detach()
        H_hat = self.supervisor(E_hat, T).detach()

        # Forward Pass
        Y_real = self.discriminator(H, T)  # Encoded original data
        Y_fake = self.discriminator(H_hat, T)  # Output of generator + supervisor
        Y_fake_e = self.discriminator(E_hat, T)  # Output of generator

        try:
            if self.gradient_type == "wasserstein":
                # TODO remove logs
                D_loss = -torch.mean(Y_real) + torch.mean(Y_fake) + torch.mean(Y_fake_e)

            elif self.gradient_type == "wasserstein_gp":
                gradient_penalty = compute_gradient_penalty(
                    self.discriminator, Y_real, Y_fake
                )
                D_loss = (
                    -torch.mean(Y_real)
                    + torch.mean(Y_fake)
                    + torch.mean(Y_fake_e)
                    + lambda_gp * gradient_penalty
                )
            elif self.gradient_type == "vanilla":
                D_loss_real = torch.nn.functional.binary_cross_entropy_with_logits(
                    Y_real, torch.ones_like(Y_real)
                )
                D_loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(
                    Y_fake, torch.zeros_like(Y_fake)
                )
                D_loss_fake_e = torch.nn.functional.binary_cross_entropy_with_logits(
                    Y_fake_e, torch.zeros_like(Y_fake_e)
                )
                D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e
            else:
                raise UnknowGANType("Unknow GAN type!")
        except Exception:
            print("Error in GAN loss evaluation")


        return D_loss

    def _generator_forward(self, X, T, Z, gamma=1):
        """The generator forward pass
        Args:
            - X: the original feature input
            - T: the temporal information
            - Z: the noise for generator input
        Returns:
            - G_loss: the generator's loss
        """
        # Supervisor Forward Pass
        H = self.embedder(X, T)
        H_hat_supervise = self.supervisor(H, T)

        # Generator Forward Pass
        E_hat = self.generator(Z, T)
        H_hat = self.supervisor(E_hat, T)

        # Synthetic data generated
        X_hat = self.recovery(H_hat, T)

        # Generator Loss
        # 1. Adversarial loss
        Y_fake = self.discriminator(H_hat, T)  # Output of supervisor
        Y_fake_e = self.discriminator(E_hat, T)  # Output of generator

        if self.gradient_type == "vanilla":
            G_loss_U = torch.nn.functional.binary_cross_entropy_with_logits(
                Y_fake, torch.ones_like(Y_fake)
            )
            G_loss_U_e = torch.nn.functional.binary_cross_entropy_with_logits(
                Y_fake_e, torch.ones_like(Y_fake_e)
            )

            # 2. Supervised loss
            G_loss_S = torch.nn.functional.mse_loss(
                H_hat_supervise[:, :-1, :], H[:, 1:, :]
            )  # Teacher forcing next output

            # 3. Two Momments
            G_loss_V1 = torch.mean(
                torch.abs(
                    torch.sqrt(X_hat.var(dim=0, unbiased=False) + 1e-6)
                    - torch.sqrt(X.var(dim=0, unbiased=False) + 1e-6)
                )
            )
            G_loss_V2 = torch.mean(torch.abs((X_hat.mean(dim=0)) - (X.mean(dim=0))))

            G_loss_V = G_loss_V1 + G_loss_V2

            # 4. Summation
            G_loss = (
                G_loss_U
                + gamma * G_loss_U_e
                + 100 * torch.sqrt(G_loss_S)
                + 100 * G_loss_V
            )



        else:  # WGAN or WGAN_GP
            G_loss = -torch.mean(Y_fake) - torch.mean(Y_fake_e)

        return G_loss

    def _inference(self, Z, T):
        """Inference for generating synthetic data
        Args:
            - Z: the input noise
            - T: the temporal information
        Returns:
            - X_hat: the generated data
        """
        # Generator Forward Pass
        E_hat = self.generator(Z, T)
        H_hat = self.supervisor(E_hat, T)

        # Synthetic data generated
        X_hat = self.recovery(H_hat, T)
        return X_hat

    def forward(self, X, T, Z, obj, gamma=1):
        """
        Args:
            - X: the input features (B, H, F)
            - T: the temporal information (B)
            - Z: the sampled noise (B, H, Z)
            - obj: the network to be trained (`autoencoder`, `supervisor`, `generator`, `discriminator`)
            - gamma: loss hyperparameter
        Returns:
            - loss: The loss for the forward pass
            - X_hat: The generated data
        """
        if obj != "inference":
            if X is None:
                raise ValueError("`X` should be given")

            X = torch.FloatTensor(X)
            X = X.to(self.device)

        if Z is not None:
            Z = torch.FloatTensor(Z)
            Z = Z.to(self.device)

        if obj == "autoencoder":
            # Embedder & Recovery
            loss = self._recovery_forward(X, T)

        elif obj == "supervisor":
            # Supervisor
            loss = self._supervisor_forward(X, T)

        elif obj == "generator":
            if Z is None:
                raise ValueError("`Z` is not given")

            # Generator
            loss = self._generator_forward(X, T, Z)

        elif obj == "discriminator":
            if Z is None:
                raise ValueError("`Z` is not given")

            # Discriminator
            loss = self._discriminator_forward(X, T, Z)

            return loss

        elif obj == "inference":

            X_hat = self._inference(Z, T)
            X_hat = X_hat.cpu().detach()

            return X_hat

        else:
            raise ValueError(
                "`obj` should be either `autoencoder`, `supervisor`, `generator`, or `discriminator`"
            )

        return loss


if __name__ == "__main__":
    pass
