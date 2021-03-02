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

import os
import itertools
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

from newords import find_new_words_with_anchor_words, NewWordDetector, SENTENCE_JOINT_CHR, NUM_CPU

# 当前目录
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_PATH = os.path.abspath(os.path.join(CUR_PATH, '../'))
DATA_PROCESSED_PATH = os.path.join(PROJECT_ROOT_PATH, 'data')


def process_origin_corpus_data_single(content: pd.DataFrame,
                                      start_idx: int,
                                      split=False,
                                      data_idx=-1,
                                      min_sentence_len=3):
    """
    对数据集进行重新整理，以文章还是句子为最小单位
    :param content:
    :param start_idx:
    :param split:
    :param data_idx:
    :param min_sentence_len:
    :param level:
    :return:
    """
    sentence_joint_chr = SENTENCE_JOINT_CHR
    result_data = []
    # 句子和其文章对应对应， sent_article_map[i] = j，表示第i个句子属于第j篇文章
    sent_article_map = []
    content.fillna(value="", inplace=True)
    data_idx = content.shape[1] - 1 if data_idx == -1 else data_idx
    for idx, line in tqdm(content.iterrows(), "processing corpus"):
        item = line[data_idx]
        for sentence in item.split(sentence_joint_chr):
            if not sentence or sentence == '\n' or sentence == '\r\n':
                continue
            if len(sentence) < min_sentence_len:
                continue
            if split:
                result_data.append(sentence.split())
            else:
                result_data.append(sentence)
            # idx并不一定从0开始，不需要加上偏移量start_idx
            sent_article_map.append(idx)

    return result_data, sent_article_map


def process_origin_corpus_data(content: pd.DataFrame,
                               split=False,
                               data_idx=-1,
                               min_sentence_len=3):
    num_task = NUM_CPU
    partial_len = int(np.ceil(len(content) / num_task))
    if partial_len == 0:
        partial_len = len(content)
        num_task = 1
    partial_results = Parallel(n_jobs=num_task, backend="multiprocessing")(
        delayed(process_origin_corpus_data_single)(content.iloc[idx:idx + partial_len],
                                                   idx,
                                                   split,
                                                   data_idx,
                                                   min_sentence_len)
        for idx in range(0, len(content), partial_len))
    print("merge the result of multi-processing")
    # 合并结果
    result_data_list = [partial[0] for partial in partial_results]
    result_data = list(itertools.chain(*result_data_list))

    sent_article_map_list = [partial[1] for partial in partial_results]
    sent_article_map = list(itertools.chain(*sent_article_map_list))
    assert len(sent_article_map) == len(result_data)

    return result_data, sent_article_map


def test_detector():
    detector1 = NewWordDetector("*")
    word_groups = [("你好", "世界"), ("你好", "nice"), ("你好", "世界"), ("你好", "nice"), ("你好", "世界", "王先生")]
    for word_group in word_groups:
        detector1.add(word_group)

    detector2 = NewWordDetector("*")
    word_groups = [("yes",), ("你好", "世界"), ("你好", "nice"), ("你好", "世界"), ("你好", "nice"), ("你好", "世界", "王先生"),
                   ("great", 'word')]
    for word_group in word_groups:
        detector2.add(word_group)

    detector1.update(detector2)


if __name__ == "__main__":
    processed_df = pd.read_csv('data/test_data.csv')
    processed_df, _ = process_origin_corpus_data(processed_df, split=True)
    result = find_new_words_with_anchor_words(processed_df)
    result_pd = pd.DataFrame(result, columns=['word', 'score', 'count'])
    print(result_pd)
    # new_word_path = os.path.join(common.DATA_PATH, "new_words_{}.csv".format("test"))
    # result_pd = pd.DataFrame(data=result, columns=["word", "score"])
    # print(result_pd)
    # result_pd.to_csv(new_word_path, index=False, encoding="utf-8")

    # 如果想要调试和选择其他的阈值，可以print result来调整
    print('#############################')
    for idx, line in result_pd.iterrows():
        if idx > 10:
            break
        print('{} ---->  {}'.format(line['word'], line['score']))
    print('#############################')
