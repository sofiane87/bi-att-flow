import re
import numpy as np

# 获取text中每个token的起始和终止索引  (start, stop)
def get_2d_spans(text, tokenss):
    spanss = []
    cur_idx = 0
    for tokens in tokenss:
        spans = []
        for token in tokens:
            if text.find(token, cur_idx) < 0:
                print(tokens)
                print("{} {} {}".format(token, cur_idx, text))
                raise Exception()
            cur_idx = text.find(token, cur_idx)
            spans.append((cur_idx, cur_idx + len(token)))
            cur_idx += len(token)
        spanss.append(spans)
    return spanss

# 获取包含start和stop这个词的词在文中的索引 (sent_id, word_id) (sent_id, word_id + 1)
def get_word_span(context, wordss, start, stop):
    spanss = get_2d_spans(context, wordss)
    idxs = []
    for sent_idx, spans in enumerate(spanss):
        for word_idx, span in enumerate(spans):
            if not (stop <= span[0] or start >= span[1]):
                idxs.append((sent_idx, word_idx))

    assert len(idxs) > 0, "{} {} {} {}".format(context, spanss, start, stop)
    return idxs[0], (idxs[-1][0], idxs[-1][1] + 1)


def get_phrase(context, wordss, span):
    """
    Obtain phrase as substring of context given start and stop indices in word level
    :param context:
    :param wordss:
    :param start: [sent_idx, word_idx]
    :param stop: [sent_idx, word_idx]
    :return:
    """
    start, stop = span
    flat_start = get_flat_idx(wordss, start)
    flat_stop = get_flat_idx(wordss, stop)
    words = sum(wordss, [])
    char_idx = 0
    char_start, char_stop = None, None
    for word_idx, word in enumerate(words):
        char_idx = context.find(word, char_idx)
        assert char_idx >= 0
        if word_idx == flat_start:
            char_start = char_idx
        char_idx += len(word)
        if word_idx == flat_stop - 1:
            char_stop = char_idx
    assert char_start is not None
    assert char_stop is not None
    return context[char_start:char_stop]


def get_flat_idx(wordss, idx):
    return sum(len(words) for words in wordss[:idx[0]]) + idx[1]

# 返回索引为idx的token的start位置
def get_word_idx(context, wordss, idx):
    spanss = get_2d_spans(context, wordss)
    return spanss[idx[0]][idx[1]][0]


def process_tokens(temp_tokens):
    tokens = []
    for token in temp_tokens:
        flag = False
        l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        # \u2013 is en-dash. Used for number to nubmer
        # l = ("-", "\u2212", "\u2014", "\u2013")
        # l = ("\u2013",)
        updated_token = [ word for word in re.split("([{}])".format("".join(l)), token) if len(word) != 0]
        tokens.extend(updated_token)
    return tokens


def get_best_span(ypi, yp2i):  # 获取一个片段（pi, pj+1），pi = max(p0, pj), socre=pi*pj最大
    max_val = 0
    best_word_span = (0, 1)
    best_sent_idx = 0
    for f, (ypif, yp2if) in enumerate(zip(ypi, yp2i)):   # [M, JX]
        argmax_j1 = 0
        for j in range(len(ypif)):  # [JX]
            val1 = ypif[argmax_j1]
            if val1 < ypif[j]:
                val1 = ypif[j]
                argmax_j1 = j

            val2 = yp2if[j]
            if val1 * val2 > max_val:
                best_word_span = (argmax_j1, j)
                best_sent_idx = f
                max_val = val1 * val2
    return ((best_sent_idx, best_word_span[0]), (best_sent_idx, best_word_span[1] + 1)), float(max_val)

def get_best_span_topk(ypi, yp2i, k):  # 获取一个片段（pi, pj+1），pi = max(p0, pj), socre=pi*pj最大
    topk_sent_word_span = list()
    
    for f, (ypif, yp2if) in enumerate(zip(ypi, yp2i)):   # [M, JX]
        argmax_j1 = 0
        for j in range(len(ypif)):  # [JX]
            val1 = ypif[argmax_j1]
            if val1 < ypif[j]:
                val1 = ypif[j]
                argmax_j1 = j

            val2 = yp2if[j]
            topk_sent_word_span.append([((f, argmax_j1), (f, j + 1)), float(val1 * val2)])
                
    topk_sent_word_span.sort(key=lambda x: x[1], reverse=True)
    k = min(k, len(topk_sent_word_span))
    return zip(*(topk_sent_word_span[:k]))

def get_best_span_topk_sent(ypi, yp2i, k):  # 获取一个片段（pi, pj+1），pi = max(p0, pj), socre=pi*pj最大; 限制在一句话中
    topk_sent_word_span = list()
    
    for f, (ypif, yp2if) in enumerate(zip(ypi, yp2i)):   # [M, JX]
        argmax_j1 = 0
        for j in range(len(ypif)):  # [JX]
            val1 = ypif[argmax_j1]
            if val1 < ypif[j]:
                val1 = ypif[j]
                argmax_j1 = j

            val2 = yp2if[j]
            topk_sent_word_span.append([((f, argmax_j1), (f, j + 1)), float(val1 * val2)])
                
    topk_sent_word_span.sort(key=lambda x: x[1], reverse=True)
    k = min(k, len(topk_sent_word_span))
    return zip(*(topk_sent_word_span[:k]))

def get_best_span_wy(wypi, th):     # 获取多个块，其中每个块有连续的单词组成，每个单词的pi都大于阈值th
    chunk_spans = []                # 如果所有的值都小于0.5，则只取最大的哪一个
    scores = []
    chunk_start = None
    score = 0
    l = 0
    th = min(th, np.max(wypi))
    for f, wypif in enumerate(wypi):    # [M, JX]
        for j, wypifj in enumerate(wypif):  # [JX]
            if wypifj >= th:
                if chunk_start is None:
                    chunk_start = f, j
                score += wypifj
                l += 1
            else:
                if chunk_start is not None:
                    chunk_stop = f, j
                    chunk_spans.append((chunk_start, chunk_stop))
                    scores.append(score/l)
                    score = 0
                    l = 0
                    chunk_start = None
        if chunk_start is not None:
            chunk_stop = f, j+1
            chunk_spans.append((chunk_start, chunk_stop))
            scores.append(score/l)
            score = 0
            l = 0
            chunk_start = None

    return max(zip(chunk_spans, scores), key=lambda pair: pair[1])


def get_span_score_pairs(ypi, yp2i):
    span_score_pairs = []
    for f, (ypif, yp2if) in enumerate(zip(ypi, yp2i)):
        for j in range(len(ypif)):
            for k in range(j, len(yp2if)):
                span = ((f, j), (f, k+1))
                score = ypif[j] * yp2if[k]
                span_score_pairs.append((span, score))
    return span_score_pairs


