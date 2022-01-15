import torch
from torch.utils.data import Dataset
import numpy as np
import random
import heapq

import config

def source2ids(source_words, vocab):
    ids = []
    oovs = []
    unk_id = vocab.UNK
    for w in source_words:
        i = vocab[w]
        if i == unk_id:  # 假如单词是OOV
            if w not in oovs:  # 假如w不在OOV列表中，就加进入列表
                oovs.append(w)
            oov_num = oovs.index(w)
            ids.append(vocab.size() + oov_num)
        else:
            ids.append(i)
    return ids, oovs

def abstract2ids(abstract_words, vocab, source_oovs):
    ids = []
    unk_id = vocab.UNK
    for w in abstract_words:
        i = vocab[w]
        if i == unk_id:  # 假如单词是OOV
            if w in source_oovs:  # 假如w在OOV列表中
                vocab_idx = vocab.size() + source_oovs.index(w)
                ids.append(vocab_idx)
            else:
                ids.append(unk_id)  # 假如不在，就映射到 UNK 的id
        else:
            ids.append(i)
    return ids

def sort_batch_by_len(data_batch):
    res = {'x': [],
           'y': [],
           'x_len': [],
           'y_len': [],
           'OOV': [],
           'len_OOV': []}
    for i in range(len(data_batch)):
        res['x'].append(data_batch[i]['x'])
        res['y'].append(data_batch[i]['y'])
        res['x_len'].append(len(data_batch[i]['x']))
        res['y_len'].append(len(data_batch[i]['y']))
        res['OOV'].append(data_batch[i]['OOV'])
        res['len_OOV'].append(data_batch[i]['len_OOV'])

    # 根据 x_len 的长度来重排
    sorted_indices = np.array(res['x_len']).argsort()[::-1].tolist()
    data_batch = {name: [_tensor[i] for i in sorted_indices]
                  for name, _tensor in res.items()}
    return data_batch

class SampleDataset(Dataset):
    def __init__(self, PairDataset, vocab):
        self.src_sents = [x[0] for x in PairDataset]
        self.trg_sents = [x[1] for x in PairDataset]
        self.vocab = vocab
        self._len = len(PairDataset)

    def __getitem__(self, index):
        x, oov = source2ids(self.src_sents[index], self.vocab)
        return {
            'x': [self.vocab.SOS] + x + [self.vocab.EOS],
            'OOV': oov,
            'len_OOV': len(oov),
            'y': [self.vocab.SOS] + abstract2ids(self.trg_sents[index], self.vocab, oov) + [self.vocab.EOS],
            'x_len': len(self.src_sents[index]),
            'y_len': len(self.trg_sents[index])}

    def __len__(self):
        return self._len

def collate_fn(batch):
    def padding(indice, max_length, pad_idx=0):
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    data_batch = sort_batch_by_len(batch)

    x = data_batch["x"]
    x_max_length = max([len(t) for t in x])
    y = data_batch["y"]
    y_max_length = max([len(t) for t in y])

    OOV = data_batch["OOV"]
    len_OOV = torch.tensor(data_batch["len_OOV"])

    x_padded = padding(x, x_max_length)
    y_padded = padding(y, y_max_length)

    x_len = torch.tensor(data_batch["x_len"])
    y_len = torch.tensor(data_batch["y_len"])
    return x_padded, y_padded, x_len, y_len, OOV, len_OOV  # 除了 OOV 是个列表外， 其它都是tensor

def replace_oovs(in_tensor, vocab):
    oov_token = torch.full(in_tensor.shape, vocab.UNK).long().to(config.device)
    out_tensor = torch.where(in_tensor > len(vocab) - 1, oov_token, in_tensor)
    return out_tensor

class ScheduledSampler():
    def __init__(self, phases):
        self.phases = phases
        self.scheduled_probs = [i / (self.phases - 1) for i in range(self.phases)]

    def teacher_forcing(self, phase):
        sampling_prob = random.random()
        if sampling_prob >= self.scheduled_probs[phase]:
            return True
        else:
            return False

def outputids2words(id_list, source_oovs, vocab):
    words = []
    for i in id_list:
        try:
            w = vocab.index2word[i]  # might be [UNK]
        except IndexError:  # w is OOV
            assert_msg = "Error: cannot find the ID the in the vocabulary."
            assert source_oovs is not None, assert_msg
            source_oov_idx = i - vocab.size()
            try:
                w = source_oovs[source_oov_idx]
            except ValueError:  # i doesn't correspond to an source oov
                raise ValueError(
                    'Error: model produced word ID %i corresponding to source OOV %i \
                     but this example only has %i source OOVs'
                    % (i, source_oov_idx, len(source_oovs)))
        words.append(w)
    return ' '.join(words)


#beamsearch辅助函数
class Beam(object):
    def __init__(self,
                 tokens,
                 log_probs,
                 decoder_states,
                 coverage_vector):
        self.tokens = tokens
        self.log_probs = log_probs
        self.decoder_states = decoder_states
        self.coverage_vector = coverage_vector

    def extend(self,
               token,
               log_prob,
               decoder_states,
               coverage_vector):
        return Beam(tokens=self.tokens + [token],
                    log_probs=self.log_probs + [log_prob],
                    decoder_states=decoder_states,
                    coverage_vector=coverage_vector)

    def seq_score(self):
        """
        This function calculate the score of the current sequence.
        The scores are calculated according to the definitions in
        https://opennmt.net/OpenNMT/translation/beam_search/.
        1. Lenth normalization is used to normalize the cumulative score
        of a whole sequence.
        2. Coverage normalization is used to favor the sequences that fully
        cover the information in the source. (In this case, it serves different
        purpose from the coverage mechanism defined in PGN.)
        3. Alpha and beta are hyperparameters that used to control the
        strengths of ln and cn.
        """
        len_Y = len(self.tokens)
        # Lenth normalization
        ln = (5 + len_Y) ** config.alpha / (5 + 1) ** config.alpha
        cn = config.beta * torch.sum(  # Coverage normalization
            torch.log(
                config.eps +
                torch.where(
                    self.coverage_vector < 1.0,
                    self.coverage_vector,
                    torch.ones((1, self.coverage_vector.shape[1])).to(torch.device(config.device))
                )
            )
        )

        score = sum(self.log_probs) / ln + cn
        return score
    def __lt__(self, other):
        return self.seq_score() < other.seq_score()
    def __le__(self, other):
        return self.seq_score() <= other.seq_score()

def add2heap(heap, item, k):
    if len(heap) < k:
        heapq.heappush(heap, item)
    else:
        heapq.heappushpop(heap, item)

