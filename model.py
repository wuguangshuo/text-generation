import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from utils import replace_oovs


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size,):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded)
        return output, hidden

class Attention(nn.Module):
    def __init__(self, hidden_units):
        super(Attention, self).__init__()
        self.Wh = nn.Linear(2*hidden_units, 2*hidden_units, bias=False)
        self.Ws = nn.Linear(2*hidden_units, 2*hidden_units)
        self.wc = nn.Linear(1, 2*hidden_units, bias=False)
        self.v = nn.Linear(2*hidden_units, 1, bias=False)

    def forward(self, decoder_states, encoder_output, x_padding_masks, coverage_vector):
        """
        Args:
            decoder_states (tuple): each with shape (1, batch_size, hidden_units)
            encoder_output (Tensor): shape (batch_size, seq_len, hidden_units).
            x_padding_masks (Tensor): shape (batch_size, seq_len).
            coverage_vector (Tensor): shape (batch_size, seq_len).

        Returns:
            context_vector (Tensor): shape (batch_size, 2*hidden_units).
            attention_weights (Tensor): shape (batch_size, seq_length).
            coverage_vector (Tensor): shape (batch_size, seq_length).
        """
        # 把 decoder 的 hidden 向量和 ceil 向量拼接起来作为状态向量
        h_dec, c_dec = decoder_states#[1,b,h]
        s_t = torch.cat([h_dec, c_dec], dim=2)#[1,b,2h]  # (1, batch_size, 2*hidden_units)
        s_t = s_t.transpose(0, 1)  # (batch_size, 1, 2*hidden_units)
        s_t = s_t.expand_as(encoder_output).contiguous()  # (batch_size, seq_length, 2*hidden_units)

        # 计算attention
        encoder_features = self.Wh(encoder_output.contiguous())  #  (batch_size, seq_length, 2*hidden_units)
        decoder_features = self.Ws(s_t)  # (batch_size, seq_length, 2*hidden_units)
        att_inputs = encoder_features + decoder_features  # (batch_size, seq_length, 2*hidden_units)
        # 增加 coverage 向量.
        if config.coverage:
            coverage_features = self.wc(coverage_vector.unsqueeze(2))#(batch_size, seq_length, 2*hidden_units)
            att_inputs = att_inputs + coverage_features#(batch_size, seq_length, 2*hidden_units)

        # 求attention概率分布
        score = self.v(torch.tanh(att_inputs))  # (batch_size, seq_length, 1)
        attention_weights = F.softmax(score, dim=1).squeeze(2)  # (batch_size, seq_length)
        attention_weights = attention_weights * x_padding_masks#填充部位的注意力变为0
        normalization_factor = attention_weights.sum(1, keepdim=True)  # Normalize attention weights after excluding padded positions.
        attention_weights = attention_weights / normalization_factor#注意力归一化[b,s]
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_output)  # (batch_size, 1, 2*hidden_units)
        context_vector = context_vector.squeeze(1)  # (batch_size, 2*hidden_units)

        # Update coverage vector.
        if config.coverage:
            coverage_vector = coverage_vector + attention_weights#更新覆盖向量,用先前的注意力权重决策来影响当前注意力权重的决策，这样就避免在同一位置重复，从而避免重复生成文本。

        return context_vector, attention_weights, coverage_vector# (batch_size, 2*hidden_units),[b,s],[b,s]

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, enc_hidden_size=None, is_cuda=True):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.device = torch.device('cuda') if is_cuda else torch.device('cpu')
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

        self.W1 = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.W2 = nn.Linear(self.hidden_size, vocab_size)
        if config.pointer:
            self.w_gen = nn.Linear(self.hidden_size * 4 + embed_size, 1)

    def forward(self, x_t, decoder_states, context_vector):
        """
        Args:
            x_t (Tensor): shape (batch_size, 1).
            decoder_states (tuple): (h_n, c_n), each with shape (1, batch_size, hidden_units) for each.
            context_vector (Tensor): shape (batch_size,2*hidden_units).
        Returns:
            p_vocab (Tensor): shape (batch_size, vocab_size).
            docoder_states (tuple): The lstm states in the decoder.Each with shapes (1, batch_size, hidden_units).
            p_gen (Tensor): shape (batch_size, 1).
        """
        decoder_emb = self.embedding(x_t)#解码器当前时刻输入[b,1,h]
        decoder_output, decoder_states = self.lstm(decoder_emb, decoder_states)#[b,1,h]  decoder_states中h,c为[1,b,h]

        # 拼接 状态向量 和 上下文向量
        decoder_output = decoder_output.view(-1, config.hidden_size)#[b,h]
        concat_vector = torch.cat([decoder_output, context_vector], dim=-1)  # (batch_size, 3*hidden_units)

        #
        FF1_out = self.W1(concat_vector)  # (batch_size, hidden_units)
        FF2_out = self.W2(FF1_out)  # (batch_size, vocab_size)
        p_vocab = F.softmax(FF2_out, dim=1)  # (batch_size, vocab_size)

        # Concatenate h and c to get s_t and expand the dim of s_t.
        h_dec, c_dec = decoder_states#更新隐状态准备下一时刻使用
        s_t = torch.cat([h_dec, c_dec], dim=2)  # (1, batch_size, 2*hidden_units)

        p_gen = None
        if config.pointer:
            x_gen = torch.cat([context_vector, s_t.squeeze(0), decoder_emb.squeeze(1)], dim=-1)
            p_gen = torch.sigmoid(self.w_gen(x_gen))
        return p_vocab, decoder_states, p_gen

class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

    def forward(self, hidden):
        h, c = hidden#(num_layers * num_directions, batch, hidden_size)
        h_reduced = torch.sum(h, dim=0, keepdim=True)
        c_reduced = torch.sum(c, dim=0, keepdim=True)
        return (h_reduced, c_reduced)


class PGN(nn.Module):
    def __init__(self, v):
        super(PGN, self).__init__()
        self.v = v
        self.device = config.device
        self.attention = Attention(config.hidden_size)
        self.encoder = Encoder(len(v), config.embed_size, config.hidden_size)
        self.decoder = Decoder(len(v), config.embed_size, config.hidden_size)
        self.reduce_state = ReduceState()

    def get_final_distribution(self, x, p_gen, p_vocab, attention_weights, max_oov):
        """
        Args:
            x: (batch_size, seq_len)
            p_gen: (batch_size, 1)
            p_vocab: (batch_size, vocab_size)
            attention_weights: (batch_size, seq_len)
            max_oov: (Tensor or int): The maximum sequence length in the batch.
        Returns:
            final_distribution (Tensor): shape (batch_size, )
        """

        if not config.pointer:
            return p_vocab

        batch_size = x.size()[0]
        # Clip the probabilities.
        p_gen = torch.clamp(p_gen, 0.001, 0.999)
        p_vocab_weighted = p_gen * p_vocab  # Get the weighted probabilities.
        attention_weighted = (1 - p_gen) * attention_weights  # (batch_size, seq_len)

        # 得到 词典 和 oov 的总体概率分布
        # extended_size = len(self.v) + max_oovs
        extension = torch.zeros((batch_size, max_oov)).float().to(self.device)
        p_vocab_extended = torch.cat([p_vocab_weighted, extension], dim=1)  # (batch_size, extended_vocab_size)

        #
        final_distribution = p_vocab_extended.scatter_add_(dim=1, index=x, src=attention_weighted)

        return final_distribution

#     @timer('model forward')
    def forward(self, x, x_len, y, len_oovs, batch, num_batches, teacher_forcing):
        """
        Args:
            x (Tensor): shape (batch_size, seq_len)
            x_len ([int): Sequence length of the current batch.
            y (Tensor): shape (bacth_size, y_len)
            len_oovs (Tensor)
            batch (int)
            num_batches(int)
            teacher_forcing(bool)
        """
        x_copy = replace_oovs(x, self.v)#在abstract2ids与source2ids期间，将oov单词变为了大于词表大小的数字，所以需要将输入全都变到词表里的UNK，防止出现embedding错误
        x_padding_masks = torch.ne(x, 0).float()
        encoder_output, encoder_states = self.encoder(x_copy)#encoder_output[b,s,h*num_layer] encoder_states包括h,c[(num_layers * num_directions, batch, hidden_size]
        decoder_states = self.reduce_state(encoder_states)#将h,c变为[1,b,h]
        coverage_vector = torch.zeros(x.size()).to(self.device)#覆盖向量初始全为0 [b,s]
        step_losses = []
        x_t = y[:, 0]#初始输入sos,解码器的初始输入
        for t in range(y.shape[1]-1):
            if teacher_forcing:#是否以上一个时刻输出为输入
                x_t = y[:, t]
            x_t = replace_oovs(x_t, self.v)#
            y_t = y[:, t+1]#标签
            #上一刻隐状态与编码器做注意力获得上下文向量context_vector(batch_size, 2 * hidden_units)，得到抽取的分布attention_weights[b, s]，更新覆盖向量coverage_vector[b, s]
            context_vector, attention_weights, coverage_vector = self.attention(decoder_states, encoder_output, x_padding_masks, coverage_vector)
            #获得生成概率p_vocab[b,vocab_size],更新ecoder_states，获得生成概率
            p_vocab, decoder_states, p_gen = self.decoder(x_t.unsqueeze(1), decoder_states, context_vector)
            #获得最终分布
            final_dist = self.get_final_distribution(x, p_gen, p_vocab, attention_weights, torch.max(len_oovs))
            x_t = torch.argmax(final_dist, dim=1).to(self.device)
            if not config.pointer:
                y_t = replace_oovs(y_t, self.v)
            target_probs = torch.gather(final_dist, 1, y_t.unsqueeze(1))#收集输入的特定维度指定位置的数值，就是真实标签在预测中的概率
            target_probs = target_probs.squeeze(1)
            mask = torch.ne(y_t, 0).float()
            loss = -torch.log(target_probs + config.eps)#Do smoothing to prevent getting NaN loss because of log(0) target_probs越大，预测越准确，损失函数需要极小化，故加负号
            if config.coverage:
                ct_min = torch.min(attention_weights, coverage_vector)
                cov_loss = torch.sum(ct_min, dim=1)
                loss = loss + config.LAMBDA * cov_loss
            mask = mask.float()
            loss = loss * mask
            step_losses.append(loss)

        sample_losses = torch.sum(torch.stack(step_losses, 1), 1)#沿着一个新维度对输入张量序列进行连接
        seq_len_mask = torch.ne(y, 0).float()
        batch_seq_len = torch.sum(seq_len_mask, dim=1)#真实摘要长度
        batch_loss = torch.mean(sample_losses / batch_seq_len)
        return batch_loss
    def load_model(self):
        # if (os.path.exists(config.output_dir)):
        #     print('Loading model: ', config.output_dir)
            # self.model = torch.load('./save_model/best_model.pkl')
        if (os.path.exists(config.encoder_save_name)):
            print('Loading model: ', config.encoder_save_name)
            self.encoder = torch.load(config.encoder_save_name)
            self.decoder = torch.load(config.decoder_save_name)
            self.attention = torch.load(config.attention_save_name)
            self.reduce_state = torch.load(config.reduce_state_save_name)