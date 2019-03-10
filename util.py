import numpy as np
import random
import re
import os

_PAD = b"<pad>"
_SOS = b"<sos>"
_EOS = b"<eos>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _SOS, _EOS, _UNK]

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3


_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")


def tokenize(string):
    return [int(s) for s in string.split()]


def pair_iter(fnamex, fnamey, batch_size, num_layers, sort_and_shuffle=True, max_seq_len=100):
    fdx, fdy = open(fnamex), open(fnamey)
    batches = []

    while True:
        # batches == 0 ,代表从最开始进去，读一行
        if len(batches) == 0:
            refill(batches, fdx, fdy, batch_size, max_seq_len, sort_and_shuffle=sort_and_shuffle)
        if len(batches) == 0:
            break
        # 数据： 标签
        x_tokens, y_tokens = batches.pop(0)
        y_tokens = add_sos_eos(y_tokens)
        x_padded, y_padded = padded(x_tokens), padded(y_tokens)
        # 转置
        source_tokens = np.array(x_padded).T
        # boolen 转成 int32 ? todo dataframe: 只对其中满足判断条件的进行该操作，单独boolean
        # ASCII 转为数字掩码
        source_mask = (source_tokens != PAD_ID).astype(np.int32)
        target_tokens = np.array(y_padded).T
        target_mask = (target_tokens != PAD_ID).astype(np.int32)

        # 源数据，掩码源数据，标签，掩码标签
        yield (source_tokens, source_mask, target_tokens, target_mask)

    return

# 给Batches发给数据
def refill(batches, fdx, fdy, batch_size, max_seq_len, sort_and_shuffle=True):
    line_pairs = []
    linex, liney = fdx.readline(), fdy.readline()
    # 如果x是空格，继续读
    while linex and liney:
        if len(linex.strip()) == 0:
            linex, liney = fdx.readline(), fdy.readline()
            continue
        x_tokens, y_tokens = tokenize(linex), tokenize(liney)

        if len(x_tokens) < max_seq_len and len(y_tokens) < max_seq_len:
            line_pairs.append((x_tokens, y_tokens))

        linex, liney = fdx.readline(), fdy.readline()
    # 随机打乱
    if sort_and_shuffle:
        random.shuffle(line_pairs)
        # e[0]：对x进行排序
        line_pairs = sorted(line_pairs, key=lambda e: len(e[0]))

    # xrange,省内存
    for batch_start in xrange(0, len(line_pairs), batch_size):
        x_batch, y_batch = zip(*line_pairs[batch_start:batch_start+batch_size])
        batches.append((x_batch, y_batch))

    if sort_and_shuffle:
        random.shuffle(batches)
    return


def add_sos_eos(tokens):
    return map(lambda token_list: [SOS_ID] + token_list + [EOS_ID], tokens)

# 填充作用，把短的句子后面填充为0，变为统一长度
def padded(tokens):
    len_toks = [len(sent) for sent in tokens]
    maxlen = max(len_toks)
    return map(lambda token_list, cur_len: token_list + [PAD_ID] * (maxlen - cur_len), tokens, len_toks)


def read_vocab(path_vocab):
    if os.path.exists(path_vocab):
        rev_vocab = []
        for line in file(path_vocab):
            rev_vocab.append(line.strip('\n'))
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", path_vocab)

def detokenize(sentence, rev_vocab):
    return ''.join([rev_vocab[ele] for ele in sentence])

def remove_nonascii(text):
    return re.sub(r'[^\x00-\x7F]', '', text)

def sentenc_to_token_ids(sentence, vocab, flag_ascii):
    if flag_ascii:
        sentence = remove_nonascii(sentence)
    return [vocab.get(ch, UNK_ID) for ch in list(sentence)]