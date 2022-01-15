from collections import Counter

from vocab import Vocab
import config

def simple_tokenizer(text):
    return text.split()

def count_words(counter, text):
    for sentence in text:
        for word in sentence:
            counter[word] += 1

class PairDataset(object):
    def __init__(self,
                 filename,
                 tokenize=simple_tokenizer,
                 max_src_len=None,
                 max_tgt_len=None,
                 truncate_src=False,
                 truncate_tgt=False):
        print("Reading dataset %s..." % filename, end=' ', flush=True)
        self.filename = filename
        self.pairs = []
        with open(filename, 'rt', encoding='utf-8') as f:
            next(f)
            for i, line in enumerate(f):
                if i<20000:
                    pair = line.strip().split('<sep>')
                    src = tokenize(pair[0])
                    if max_src_len and len(src) > max_src_len:
                        if truncate_src:
                            src = src[:max_src_len]
                        else:
                            continue
                    tgt = tokenize(pair[1])
                    if max_tgt_len and len(tgt) > max_tgt_len:
                        if truncate_tgt:
                            tgt = tgt[:max_tgt_len]
                        else:
                            continue
                    self.pairs.append((src, tgt))
                else:
                    break
        print("一共有 %d 对 pairs." % len(self.pairs))

    def build_vocab(self):
        word_counts = Counter()
        count_words(word_counts, [src + tgr for src, tgr in self.pairs])
        vocab = Vocab()
        for word, count in word_counts.most_common(config.max_vocab_size):
            vocab.add_words([word])
        return vocab

