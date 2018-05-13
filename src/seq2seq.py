import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from Models.lstm_diversity import DiverseLSTMCell

class Encoder(nn.Module):
    def __init__(self, input_size, embeddings, embed_size, hidden_size,
                 n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(input_size, embed_size)
        self.gru = nn.LSTM(embed_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)
        self.initialize(embeddings)

    def initialize(self, embeddings):
        # print(embeddings.shape, self.input_size)
        assert (embeddings.shape[0] == self.input_size)
        self.embed.weight = nn.Parameter(embeddings)

    def forward(self, src, hidden=None):
        embedded = self.embed(src)
        outputs, hidden = self.gru(embedded, hidden)
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size, num):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * num, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies).unsqueeze(1) #torch upgrade - add dim=1

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    def __init__(self, embed_size, embeddings, hidden_size, output_size,
                 n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.query_attention = Attention(hidden_size, 2)
        self.doc_attention = Attention(hidden_size, 3)
        self.gru = nn.LSTM(hidden_size + embed_size, hidden_size,
                          n_layers, dropout=dropout)
        self.distract_hard_lstm = DiverseLSTMCell(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size * 2, output_size)
        self.initialize(embeddings)

    def initialize(self, embeddings):
        assert (embeddings.size()[1] == self.embed_size)
        self.embed.weight = nn.Parameter(embeddings)

    def forward(self, input, last_hidden, distract_hidden, query_outputs, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        embedded = self.dropout(embedded)
        # calculate query attention
        query_attn_weights = self.query_attention(last_hidden[0][-1], query_outputs)
        query_context = query_attn_weights.bmm(query_outputs.transpose(0, 1))  # (B,1,N)
        query_context = query_context.transpose(0, 1)  # (1,B,N)
        
        # Calculate attention weights and apply to encoder outputs
        doc_attn_weights = self.doc_attention(torch.cat([last_hidden[0][-1].view(1, -1, self.hidden_size),\
                                              query_context], 2), encoder_outputs)
        doc_context = doc_attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        doc_context = doc_context.transpose(0, 1)  # (1,B,N)

        #Calculate Distracted document context
        distract_h, distract_c = self.distract_hard_lstm(doc_context, distract_hidden) 
        distract_hidden = (distract_h, distract_c)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, doc_context], 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        doc_context = doc_context.squeeze(0)
        output = self.out(torch.cat([output, doc_context], 1))
        output = F.log_softmax(output)
        return output, hidden, distract_hidden, doc_attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, content_encoder, query_encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.content_encoder = content_encoder
        self.decoder = decoder
        self.query_encoder = query_encoder

    def forward(self, content_src, query_src, trg, batch_size, max_len_target, teacher_forcing_ratio=0.5):
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len_target, batch_size, vocab_size)) #.cuda()

        encoder_output, hidden = self.content_encoder(content_src)
        query_output, query_hidden = self.query_encoder(query_src)
        hidden = hidden
        distract_hidden = hidden
        output = Variable(trg[0, :])  # sos
        for t in range(1, max_len_target):
            output, hidden, distract_hidden, attn_weights = self.decoder(
                    output, hidden, distract_hidden, query_output, encoder_output)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = Variable(trg[t] if is_teacher else top1) #.cuda()
        return outputs
