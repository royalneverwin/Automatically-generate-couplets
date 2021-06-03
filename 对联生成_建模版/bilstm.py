import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import random

class Encoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim, vocab_size, num_layers=1,dropout=0):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers  # RNN层数
        self.dropout = dropout  # dropout
        self.emb=nn.Embedding(self.vocab_size,self.emb_dim)
        self.rnn=nn.GRU(self.emb_dim, self.hidden_dim, num_layers=num_layers,dropout=0 )#输入为len,batch ，emb_dim
        self.dropout=nn.Dropout(dropout)
    def forward(self, input_ids,emb_matrix, lengths):
        emb=nn.Embedding.from_pretrained(emb_matrix)
        emb_in=emb(input_ids)
        packed = torch.nn.utils.rnn.pack_padded_sequence(emb_in,lengths)  # 将输入转换成torch中的pack格式，使得RNN输入的是真实长度的句子而非padding后的
        outputs, hidden = self.rnn(packed)   # [len,batch，hidden_size]
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)

        return outputs, hidden  # outputs:[len,batch,num_layer*hidden_size]  hidden:[num_layers,batch,hidden_size]


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(self.hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        self.v.data.normal_(mean=0, std=1. / np.sqrt(self.v.size(0)))

    def forward(self, hidden, encoder_outputs):
        #  encoder_outputs:(seq_len, batch_size, hidden_size)
        #  hidden:(num_layers * num_directions, batch_size, hidden_size)
        max_len = encoder_outputs.size(0)
        h = hidden[-1].repeat(max_len, 1, 1)
        # (seq_len, batch_size, hidden_size)
        attn_energies = self.score(h, encoder_outputs)  # compute attention score
        return F.softmax(attn_energies, dim=1)  # normalize with softmax

    def score(self, hidden, encoder_outputs):
        # (seq_len, batch_size, 2*hidden_size)-> (seq_len, batch_size, hidden_size)
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.permute(1, 2, 0)  # (batch_size, hidden_size, seq_len)
        v = self.v.repeat(encoder_outputs.size(1), 1).unsqueeze(1)  # (batch_size, 1, hidden_size)
        energy = torch.bmm(v, energy)  # (batch_size, 1, seq_len)
        return energy.squeeze(1)  # (batch_size, seq_len)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0):
        super().__init__()

        self.embedding_dim = embedding_dim  # 编码维度
        self.hid_dim = hidden_dim  # RNN隐层单元数
        self.vocab_size = vocab_size  # 词袋大小
        self.num_layers = num_layers  # RNN层数
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = Attention(hidden_dim)
        self.rnn = nn.GRU(embedding_dim + hidden_dim, hidden_dim,
                          num_layers=num_layers, dropout=dropout)
        self.out = nn.Linear(embedding_dim + hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input [batch_size]
        # hidden=[nlayers*dir,batchsize,hid_dim]
        # encoder_output =[len,batch,hid_dim*dir]
        input=input.unsqueeze(0) # 竖着的bsz维
        embedded=self.dropout(self.embedding(input)) # [1,bsz,emb_dim]  每个时间步
        attn_weight = self.attention(hidden, encoder_outputs)
        # (batch_size, seq_len)
        context = attn_weight.unsqueeze(1).bmm(encoder_outputs.transpose(0, 1)).transpose(0, 1)
        # (1, batch_size, hidden_dim * n_directions)
        emb_con = torch.cat((embedded, context), dim=2)
        # emb_con = [1, bsz, emb dim + hid dim]
        _, hidden = self.rnn(emb_con, hidden)
        # outputs = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        output = torch.cat((embedded.squeeze(0), hidden[-1], context.squeeze(0)), dim=1)
        output = self.out(output)
        return output, hidden, attn_weight


class Net(nn.Module):
    def __init__(self, encoder, decoder, device, teacher_forcing_ratio=0.5):
        super().__init__()

        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.loss=CrossEntropyLoss()

    def forward(self, input_ids,emb_matrix, src_lengths, trg_seqs, trg_seqs_flatten,mask): #想办法数据处理时把trg 转成 [len,batch,vocab_size]
        # src_seqs = [sent len, batch size]
        # trg_seqs = [sent len, batch size]
        batch_size = input_ids.shape[1]
        max_len = trg_seqs.shape[0]
        trg_vocab_size = self.decoder.vocab_size
        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        # hidden used as the initial hidden state of the decoder
        # encoder_outputs used to compute context
        encoder_outputs, hidden = self.encoder(input_ids,emb_matrix, src_lengths)
        # first input to the decoder is the <sos> tokens
        output = trg_seqs[0, :]
        # print("len",max_len,"net.output:",output.size(),output)

        for t in range(1, max_len):  # skip sos
            output, hidden, _ = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < self.teacher_forcing_ratio
            # print("output",output)
            # print("233",output.max(1))
            # print("hhh",output.max(1)[1])
            output = (trg_seqs[t] if teacher_force else output.max(1)[1])
        return outputs

    def predict(self, src_seqs,emb_matrix, src_lengths, max_trg_len=30, start_ix=1):
        max_src_len = src_seqs.shape[0]
        batch_size = src_seqs.shape[1]
        trg_vocab_size = self.decoder.vocab_size
        outputs = torch.zeros(max_trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src_seqs, emb_matrix,src_lengths)
        output = torch.LongTensor([4] * batch_size).to(self.device)
        attn_weights = torch.zeros((max_trg_len, batch_size, max_src_len))
        for t in range(1, max_trg_len):
            output, hidden, attn_weight = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            output = output.max(1)[1]
            # attn_weights[t] = attn_weight
        return outputs, attn_weights




