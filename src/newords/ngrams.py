# -*- coding: utf-8 -*-
# MIT License

# Copyright (c) 2021 Brokenwind

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from collections import Counter

import numpy as np
from joblib import Parallel, delayed
from nlpyutil import Logger
from tqdm import tqdm

from . import NUM_CPU

_logger = Logger()


class WordStr(str):
    """
    主要是为str加上is_group标签，标记其是否是多个词拼接成的一个词.

    注意事项:
    对WordStr进行操作时，除了add, upper, lower,其它方法都有可能导致返回结果是str，而不是WordStr，导致is_group标签丢失。
    """

    def __new__(cls, content, is_group=False):
        return super().__new__(cls, content)

    def __init__(self, content, is_group=False):
        self.is_group = is_group

    def __add__(self, other):
        return WordStr(super().__add__(other))

    def upper(self):
        return WordStr(super().upper(), self.is_group)

    def lower(self):
        return WordStr(super().lower(), self.is_group)


def generate_ngram(input_list, n, joint=False, joint_chr="_"):
    result = []
    for i in range(1, n + 1):
        result.extend(zip(*[input_list[j:] for j in range(i)]))
    if not joint:
        return result
    else:
        # 单独的词，标记为is_group=False
        joint_result = [WordStr(item[0], False) for item in result[0:len(input_list)]]
        # 拼接在一起的词，标记为is_group=True
        joint_result.extend([WordStr(joint_chr.join(item), True) for item in result[len(input_list):]])
        return joint_result


def count_ngrams_single(data_partial):
    ngram_counter = Counter()
    for word_list in tqdm(data_partial, desc="counting ngrams"):
        ngrams = generate_ngram(word_list, 3, joint=False)
        ngram_counter.update(ngrams)

    return ngram_counter


def count_ngrams(data: list):
    total_counter = Counter()
    if len(data) == 0:
        return total_counter
    num_task = NUM_CPU
    partial_len = int(np.ceil(len(data) / num_task))
    if partial_len == 0:
        partial_len = len(data)
        num_task = 1
    partial_counters = Parallel(n_jobs=num_task, backend="multiprocessing")(
        delayed(count_ngrams_single)(data[idx:idx + partial_len])
        for idx in range(0, len(data), partial_len))
    for counter in partial_counters:
        total_counter.update(counter)

    return total_counter
