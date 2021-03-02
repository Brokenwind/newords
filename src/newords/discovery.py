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

import gc
import math
import pickle
from typing import Union
from nlpyutil import Logger
from tqdm import tqdm
from . import MIN_GROUP_COUNT
from .ngrams import count_ngrams

_logger = Logger()
_FILTER_WORD_GROUPS = []


class Node:
    """
    建立字典树的节点
    """

    def __init__(self, word):
        self.word = word
        # 记录是否完成
        self.word_finish = False
        # 用来计数
        self.count = 0
        # 用来存放节点
        self.child = {}
        # 判断是否是后缀
        self.isback = False


class NewWordDetector:
    """
    建立前缀树，并且包含统计词频，计算左右熵，计算互信息的方法
    """

    def __init__(self, node,
                 init_data=None,
                 pmi_limit=1e-5,
                 word_joint_chr="_",
                 left_right_count=3,
                 filter_word_groups=None):
        """
        初始函数，data为外部词频数据集
        """
        self.root = Node(node)
        self.pmi_limit = pmi_limit
        self.word_joint_chr = word_joint_chr
        # 单个词的总量
        self.single_word_count = 0
        # 双词的总量
        self.bi_word_count = 0
        # 三词的总量
        self.tri_word_count = 0
        # 左右词的最低个数
        self.left_right_count = left_right_count
        # 根据外部词频初始化
        self.init_with_provided_dict(init_data)

    @classmethod
    def load_from_pkl(cls, pkl_path):
        """
        从pkl文件中加载对象
        """
        with open(pkl_path, 'rb') as fb:
            return pickle.load(fb)

    def save_to_pkl(self, pkl_path):
        with open(pkl_path, 'wb') as fw:
            pickle.dump(self, fw)

    def init_with_provided_dict(self, data):
        # 根据提供的词频文件进行初始化
        if data:
            node = self.root
            for key, values in data.items():
                new_node = Node(key)
                new_node.count = int(values)
                new_node.word_finish = True
                node.child[key] = new_node
                self.single_word_count += new_node.count

    def add(self, word_group: Union[tuple, list], count=1):
        """
        添加节点，对于左熵计算时，这里采用了一个trick，用a->b<-c 来表示 cab
        具体实现是利用 self.isback 来进行判断
        :return:
        """
        word_group_len = len(word_group)
        # word_group_list的每个元素由两个部分组成： (词组，是否反向表示)
        # 只有当 word_group_len == 3的时候，才会生成 反向表示词组
        word_group_list = [(word_group, False)]
        if word_group_len == 3:
            back_word_group = (word_group[1], word_group[2], word_group[0])
            word_group_list.append((back_word_group, True))
        # 遍历词组并插入
        for word_group, back_group in word_group_list:
            # 每个词组都容根节点开始插入
            node = self.root
            for idx, cur_word in enumerate(word_group):
                cur_node = node.child.get(cur_word)
                # 在节点中找词
                if cur_node:
                    node = cur_node
                else:
                    cur_node = Node(cur_word)
                    node.child[cur_word] = cur_node
                    node = cur_node
                # 判断是否是最后一个节点
                if idx == word_group_len - 1:
                    node.count += count
                    node.word_finish = True
                    node.isback = back_group
                    self.increase_count(word_group_len, count)

    def batch_add(self, multi_word_groups: list, count=1):
        """
        添加节点，对于左熵计算时，这里采用了一个trick，用a->b<-c 来表示 cab
        具体实现是利用 self.isback 来进行判断
        """
        for word_group in multi_word_groups:
            word_group_len = len(word_group)
            # word_group_list的每个元素由两个部分组成： (词组，是否反向表示)
            # 只有当 word_group_len == 3的时候，才会生成 反向表示词组
            word_group_list = [(word_group, False)]
            if word_group_len == 3:
                back_word_group = (word_group[1], word_group[2], word_group[0])
                word_group_list.append((back_word_group, True))
            # 遍历词组并插入
            for word_group, back_group in word_group_list:
                # 每个词组都容根节点开始插入
                node = self.root
                for idx, cur_word in enumerate(word_group):
                    cur_node = node.child.get(cur_word)
                    # 在节点中找词
                    if cur_node:
                        node = cur_node
                    else:
                        cur_node = Node(cur_word)
                        node.child[cur_word] = cur_node
                        node = cur_node
                    # 判断是否是最后一个节点
                    if idx == word_group_len - 1:
                        node.count += count
                        node.word_finish = True
                        node.isback = back_group
                        self.increase_count(word_group_len, count)

    def update(self, other: "NewWordDetector"):
        """
        使用另一个NewWordDetector更新当前NewWordDetector
        """
        self_node = self.root
        other_node = other.root

        self.single_word_count += other.single_word_count
        self.bi_word_count += other.bi_word_count
        self.tri_word_count += other.tri_word_count
        self.merge_tree(self_node, other_node)

        return self

    def merge_tree(self, self_node: Node, other_node: Node):
        """
        合并两颗Trie Tree
        """
        for word in other_node.child:
            tmp_other_node = other_node.child.get(word)
            tmp_self_node = self_node.child.get(word)
            if tmp_self_node:
                tmp_self_node.count += tmp_other_node.count
                tmp_self_node.word_finish |= tmp_other_node.word_finish
                isback_equal = tmp_self_node.isback == tmp_other_node.isback
                if not isback_equal:
                    _logger.info("The isback is not match")
                self.merge_tree(tmp_self_node, tmp_other_node)
            else:
                # 如果other中存在的，self不存在，直接将other中的复制过来
                self_node.child[word] = tmp_other_node
                continue

    def release(self, node):
        """
        递归释放内存空间
        """
        for word in list(node.child.keys()):
            tmp_other_node = node.child.get(word)
            self.release(tmp_other_node)
            del node.child[word]

    def set_count(self, word_group: Union[tuple, list], isback=False, count=0):
        """
        :param word_group: 词组
        :param count: 数量
        :param isback: 是否后缀表达
        :return:
        """
        status = False
        word_group_len = len(word_group)
        if word_group_len == 3 and isback:
            word_group = (word_group[1], word_group[2], word_group[0])
        # 每个词组都容根节点开始插入
        node = self.root
        for idx, cur_word in enumerate(word_group):
            node = node.child.get(cur_word, None)
            # 在节点中找词
            if not node:
                break
            # 判断是否是最后一个节点
            if idx == word_group_len - 1 and node.word_finish and node.isback == isback:
                self.increase_count(word_group_len, count - node.count)
                node.count = count
                status = True

        return status

    def delete(self, word_group: Union[tuple, list], isback=False):
        """
        删除一个词或者词组
        :param word_group:
        :return:
            status: 删除是否成功，
            couont: 被删除词组的个数
        """
        word_group_len = len(word_group)
        word_group_str = "_".join(word_group)
        count = 0
        status = False
        # 每个词组都容根节点开始插入
        node = self.root
        for idx, cur_word in enumerate(word_group):
            node = node.child.get(cur_word)
            # 在节点中找词
            if not node:
                print("can't get the word:{} in word group:{}".format(cur_word, word_group_str))
                break
            # 判断是否是最后一个节点
            if idx == word_group_len - 1:
                if node.word_finish:
                    if isback == node.isback:
                        count = node.count
                        node.word_finish = False
                        node.count = 0
                        self.decrease_count(word_group_len, count)
                        status = True
                    else:
                        print("find the word group:{}, but isback not satisfied".format(word_group_str))
                else:
                    print("the word group:{} is not complete".format(word_group_str))

        return status, count

    def merge_bigroup(self, word_group: Union[tuple, list], delimiter="_"):
        """
        删除一个词或者词组
        :param word_group:
        :return:
            status: 删除是否成功，
            couont: 被删除词组的个数
        """
        word_group_len = len(word_group)
        assert word_group_len == 2
        word_group_str = delimiter.join(word_group)

        # 1. 处理前缀是word_group的三元词组
        node = self.root
        for idx, cur_word in enumerate(word_group):
            node = node.child.get(cur_word)
            # 在节点中找词
            if not node:
                print("can't get the word:{} in word group:{}".format(cur_word, word_group_str))
                return False
        back_words = []
        if node.word_finish:
            # 逻辑上删除当前啊2gram
            node.word_finish = False
            self.decrease_count(word_group_len, node.count)
            node.count = 0
            # 处理孩子节点
            if node.child:
                for child_key in list(node.child.keys()):
                    child = node.child.pop(child_key)
                    if child.isback:
                        # 在插入正常词序为(a,b,c)时，会同时插入(a,b,c) 和 (b,c,a) and isback=True
                        # 当前处理的是(b,c,a) and isback=True的情况，合并了bc。
                        # (a,b,c) 中的b,c也要合并，通过back_words记录a，方便后续查找并处理(a,b,c)
                        new_word_group = (child.word, word_group_str)
                        back_words.append(child.word)
                    else:
                        new_word_group = (word_group_str, child.word)
                    # 将3gram融合成2gram并插入树中
                    self.add(new_word_group, child.count)
                    # 逻辑上删除当前3gram
                    self.decrease_count(3, child.count)

        # 2. 处理后缀是word_group的三元词组
        for word in back_words:
            node = self.root.child.get(word)
            for idx, cur_word in enumerate(word_group):
                cur_node = node.child.get(cur_word)
                # 在节点中找词
                if not node:
                    break
                if idx == word_group_len - 1 and cur_node.word_finish and not cur_node.isback:
                    # 将3gram融合成2gram并插入树中
                    new_word_group = (word, word_group_str)
                    self.add(new_word_group, cur_node.count)
                    # 逻辑上删除当前3gram
                    self.decrease_count(3, cur_node.count)
                    node.child.pop(cur_word)
                node = cur_node

        return True

    def increase_count(self, group_len, count=1):
        """
        增加词组长度为group_len的计数
        :param group_len: 取值范围[1,2,3]
        :return:
        """
        if group_len == 1:
            self.single_word_count += count
        elif group_len == 2:
            self.bi_word_count += count
        else:
            self.tri_word_count += count

    def decrease_count(self, group_len, count=1):
        """
        减少词组长度为group_len的计数
        :param group_len: 取值范围[1,2,3]
        :return:
        """
        if group_len == 1:
            self.single_word_count -= count
        elif group_len == 2:
            self.bi_word_count -= count
        else:
            self.tri_word_count -= count

    def single_word_probability(self):
        """
        寻找一阶共现，并返回词概率
        :return:
        """
        result = {}
        node = self.root
        if not node.child:
            return False, 0

        for child in node.child.values():
            if child.word_finish == True:
                result[child.word] = child.count / self.single_word_count

        return result, self.single_word_count

    def left_and_right_entropy(self):
        """
        寻找右频次
        统计右熵，并返回右熵
        :return:
        """
        left_result = {}
        right_result = {}
        node = self.root
        if not node.child:
            return False, 0
        for level_1_child in tqdm(node.child.values(), desc="calculate entropy"):
            for level_2_child in level_1_child.child.values():
                cur_word = level_1_child.word + level_2_child.word
                right_total = left_total = 0
                right_entropy = left_entropy = 0.0
                # 计算cur_word的左右词个数
                for level_3_child in level_2_child.child.values():
                    if level_3_child.word_finish == True:
                        if level_3_child.isback:
                            left_total += level_3_child.count
                        else:
                            right_total += level_3_child.count

                # 过滤小词频的词
                if right_total + left_total < self.left_right_count:
                    continue

                # 计算cur_word的左右熵
                for level_3_child in level_2_child.child.values():
                    if level_3_child.word_finish == True:
                        if level_3_child.isback:
                            prob = level_3_child.count / left_total
                            left_entropy += prob * math.log(prob, 2)
                        else:
                            prob = level_3_child.count / right_total
                            right_entropy += prob * math.log(prob, 2)
                left_result[cur_word] = -left_entropy
                right_result[cur_word] = -right_entropy

        return left_result, right_result

    def bi_word_pmi(self):
        """
        寻找二阶共现，并返回log2( P(X,Y) / (P(X) * P(Y))和词概率
        :return:
        """
        result = {}
        node = self.root
        if not node.child:
            return result

        one_dict, total_one = self.single_word_probability()

        for level_1_child in tqdm(node.child.values(), desc="calculate PMI"):
            for level_2_child in level_1_child.child.values():
                if level_2_child.word_finish == True:
                    pmi = math.log(max(level_2_child.count, 1), 2) - \
                          math.log(self.bi_word_count, 2) - \
                          math.log(one_dict[level_1_child.word], 2) - \
                          math.log(one_dict[level_2_child.word], 2)
                    # 这里做了PMI阈值约束
                    if pmi > self.pmi_limit:
                        word_key = self.word_joint_chr.join([level_1_child.word, level_2_child.word])
                        result[word_key] = (pmi, level_2_child.count / self.bi_word_count, level_2_child.count)
        return result

    def find_new_words(self, score_limit=1e-5, min_count=MIN_GROUP_COUNT):
        """
        根据互信息和左右熵对新词的分进行计算，并返回所欲的新词
        :return:
        """
        _logger.info("start to calculate pmi")
        bi = self.bi_word_pmi()
        if not bi:
            return None
        _logger.info("start to calculate left and right entropy")
        # 通过搜索得到左右熵
        left_entropy, right_entropy = self.left_and_right_entropy()
        result = []
        _logger.info("start to calculate the score")
        for word_key, values in tqdm(bi.items(), desc="calculate socre"):
            word = "".join(word_key.split(self.word_joint_chr))
            # 计算公式 score = PMI + min(左熵， 右熵)
            if word not in left_entropy:
                left_entropy[word] = 0.
            if word not in right_entropy:
                right_entropy[word] = 0.
            score = (values[0] + min(left_entropy[word], right_entropy[word])) * values[1]
            # 词，成词的分数，个数
            result.append((word_key, score, values[2]))
        if not result:
            return None
        # dict 的排序, 返回是个list
        _logger.info("start to sort the list")
        result = sorted(result, key=lambda x: (x[2], x[1]), reverse=True)

        _logger.info("start to filter the list")
        result = [item for item in result if item[1] >= score_limit and item[2] >= min_count]
        filtered_result = []
        for item in result:
            item_set = set(item[0].split("_"))
            if item[1] < score_limit:
                continue
            is_filter = False
            for filter_word in _FILTER_WORD_GROUPS:
                if item_set.intersection(filter_word):
                    is_filter = True
                    break
            if not is_filter:
                filtered_result.append(item)

        _logger.info("finished the current finding")

        return filtered_result

    def merge_word_group(self, word_group: Union[str, list], delimiter='_'):
        """
        合并词组为一个词，需要更改树结构中的数据
        :return:
        """

        if isinstance(word_group, str):
            words = word_group.split(delimiter)
        else:
            words = word_group
        status, count = self.delete(words, isback=False)
        new_word_group = ("".join(words),)
        self.add(new_word_group)

        return self.set_count(new_word_group, isback=False, count=count)


def add_ngrams_to_tree(detector, data) -> NewWordDetector:
    """
    将ngrams添加到新词发现模型中，会过滤掉不包含anchor_words的句子
    """
    ngram_counter = count_ngrams(data)
    for ngram, count in tqdm(ngram_counter.items(), desc="inserting ngrams into discovery_model"):
        detector.add(ngram, count)

    return detector


def find_new_words_with_anchor_words(data: list) -> list:
    """
    根据锚定词进行新词发现
    :param data: # 二维数组, [[句子1分词list], [句子2分词list], ..., [句子n分词list]]
    """
    detector = NewWordDetector('*', None, pmi_limit=0.)
    discovery_model = add_ngrams_to_tree(detector, data)

    _logger.info("the number of single word: {}".format(discovery_model.single_word_count))
    _logger.info("the number of bi-word: {}".format(discovery_model.bi_word_count))
    _logger.info("the number of tri-word: {}".format(discovery_model.tri_word_count))

    result = discovery_model.find_new_words()

    _logger.info("release the memory of trie tree")
    del discovery_model
    gc.collect()
    _logger.info("finished releasing the memory of trie tree")

    return result
