#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2022/8/17 16:22
# @File  : myexample.py
# @Author: 
# @Desc  : 利用自己数据测试新词发现功能
import os
import glob
import json
import re
import codecs
import pandas as pd
from tqdm import tqdm
import numpy as np
from myutils import CHANNEL, query_by_channel
from word_discovery import Progress, write_corpus, count_ngrams, KenlmNgrams, filter_ngrams, SimpleTrie, filter_vocab
import logging
logging.basicConfig(level=logging.INFO, format=u'%(asctime)s - %(levelname)s - %(message)s')


def get_and_save_data(data_dir="data", start='2021-12-01', end='2022-07-31', force_download=False):
    """
    从数据库获取数据，保存到本地目录，按渠道名字分类目录
    :return:
    """
    date_df = pd.date_range(start,end, freq='M').strftime('%Y-%m-%d')
    # 然后把日期变字符串格式
    date_list = [str(date) for date in date_df]
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    for channel in Progress(CHANNEL, desc="获取数据"):
        chanel_dir = os.path.join(data_dir, channel)
        if not os.path.exists(chanel_dir):
            os.mkdir(chanel_dir)
        # 按月份获取所有数据，保存到每月的文件中
        for idx, date in enumerate(date_list):
            start_date = date
            end_date = date_list[idx + 1] if idx < len(date_list) - 1 else date
            if start_date == end_date:
                # 如果是同一天了，就不用继续了
                continue
            # 保存成text文件
            text_file = os.path.join(chanel_dir, f"{start_date}_{end_date}.csv")
            if os.path.exists(text_file) and not force_download:
                continue
            df_data = query_by_channel(channel, start_date, end_date, limit=-1)
            df_data.to_csv(text_file, sep="\t", index=False, header=False, escapechar='|')
    return date_list

def text_generator(data_dir="data", use_lines=False):
    """
    如果txt是一个大文件，那么请使用use_lines=True，如果是小文件，那么不介意了，
    因为如果是大文件，文本过长，SimpleTrie的tokenize函数会循环过长，导致循环缓慢
    param: use_lines: 每次返回是否使用是每行文件，还是一个单独的文件的所有内容
    """
    txts = glob.glob(f'{data_dir}/*.csv')
    assert len(txts) > 0, "没有找到数据文件, 请检查文件"
    for txt in tqdm(txts, desc="读取txts数据"):
        d = codecs.open(txt, encoding='utf-8').read()
        d = d.replace(u'\u3000', ' ').strip()
        data = re.sub(u'[^\u4e00-\u9fa50-9a-zA-Z ]+', '\n', d)
        if use_lines:
            for line in data.split('\n'):
                yield line
        else:
            yield data
def my_word_discovery(data_dir="data/tmall", save_dir="save", use_cache=True, min_count=32, order=4, memory=0.6):
    """
    利用自己数据测试新词发现功能
    :param save_dir: 保存词典的目录
    :param use_cache:
    :param min_count: 单词最少出现的次数
    :param order:
    :param memory: 使用kenlm时，内存使用的百分比，默认是0.5，即使用50%的内存
    """
    corpus_file = os.path.join(save_dir, 'mydata.corpus')  # 语料保存的文件名
    char_file = os.path.join(save_dir, 'mydata.chars')  # 字符集保存的文件名
    ngram_file = os.path.join(save_dir, 'mydata.ngrams')  # ngram集保存的文件名
    output_file = os.path.join(save_dir, 'mydata.vocab')  # 最后导出的词表文件名
    candidates_file = os.path.join(save_dir, 'candidates.json')  # 候选的单词词表, 保存词和词频的字典
    if use_cache and os.path.exists(corpus_file):
        logging.warning(f"注意：使用的缓存的语料库{corpus_file}，如果原始数据更新，请设置use_cache=False")
    else:
        logging.info(f"开始构建语料库，保存至{corpus_file}")
        write_corpus(text_generator(data_dir), corpus_file)  # 将语料转存为文本
    if use_cache and os.path.exists(char_file) and os.path.exists(ngram_file):
        logging.warning(f"注意：使用的缓存的词表字符{char_file}和ngram文件{ngram_file}，如果原始数据更新，请设置use_cache=False")
    else:
        logging.info(f"开始计算ngram，保存至{ngram_file}")
        count_ngrams(corpus_file, order, char_file, ngram_file, memory)  # 用Kenlm统计ngram
    logging.info(f"加载由kelm构建好的ngram")
    Kngrams = KenlmNgrams(char_file, ngram_file, order, min_count)  # 加载ngram
    logging.info(f"互信息过滤ngrams")
    output_ngrams = filter_ngrams(Kngrams.ngrams, Kngrams.total, [0, 2, 4, 6])  # 过滤ngram
    if use_cache and os.path.exists(candidates_file):
        logging.warning(f"注意：使用的缓存的候选词表{candidates_file}，如果原始数据更新，请设置use_cache=False")
        with open(candidates_file, 'r') as f:
            candidates = json.load(f)
    else:
        ngtrie = SimpleTrie()  # 构建ngram的Trie树
        logging.info(f"构建ngram的Trie树")
        for w in Progress(output_ngrams, 10000, desc=u'构建 ngram trie'):
            # w 是每个词， eg: 'B5修复'
            ngtrie.add_word(w)
        logging.info(f"最终ngram trie 构建的字典数量是: {len(ngtrie.dic)} 个")
        logging.info(f"开始发现新词，预计耗时很久，很久")
        candidates = {}  # 得到候选词
        for t in Progress(text_generator(data_dir,use_lines=True), 2, desc='发现新词中'):
            if len(candidates) % 1000:
                logging.info(f"已经发现{len(candidates)}个候选新词")
            for w in ngtrie.tokenize(t):  # 预分词
                candidates[w] = candidates.get(w, 0) + 1
        logging.info(f"最终得到的候选新词数量是: {len(candidates)} 个")
        # 包含候选词的字典保存到文件
        with open(candidates_file, 'w', encoding='utf-8') as f:
            json.dump(candidates, f)
    # 频数过滤
    candidates = {i: j for i, j in candidates.items() if j >= min_count}
    # 互信息过滤(回溯)
    logging.info(f"互信息过滤词表")
    new_words = filter_vocab(candidates, output_ngrams, order)

    # 输出结果文件
    logging.info(f"生成最终的词表文件{output_file}, 共{len(new_words)}个新词")
    with codecs.open(output_file, 'w', encoding='utf-8') as f:
        # 按词频排序保存
        for i, j in sorted(new_words.items(), key=lambda s: -s[1]):
            s = '%s %s\n' % (i, j)
            f.write(s)

def read_vocab_file(vocab_file):
    """
    读取词表文件
    :param vocab_file: 词表文件
    :return: 词表字典, 返回词表字典格式：{'word': count}
    """
    vocab_dict = {}
    assert os.path.exists(vocab_file), f"词表文件{vocab_file}不存在"
    with codecs.open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_split = line.strip().split()
            if len(line_split) == 2:
                word, count = line_split
                vocab_dict[word] = int(count)
    print(f"词表文件{vocab_file}读取完成，共{len(vocab_dict)}个词")
    return vocab_dict

def diff_two_vocab(one, two="save/universal.vocab", same_file="save/same.vocab", diff_one_file="save/diff_one.vocab", diff_two_file="save/diff_two.vocab"):
    """
    比较2个vocab字典的相同的词和不同的词
    :param one: vocab字典1, 领域1的词典
    :param two: vocab字典2， 领域2的词典，也可以用通用词典
    :param same_file: 相同的词文件
    :param diff_one_file: 不同的词文件1，即领域1内特有的词了
    :param diff_two_file: 不同的词文件2，即领域2内特有的词了，或者说词典内特有的词
    """
    vocab_one = read_vocab_file(vocab_file=one)
    vocab_two = read_vocab_file(vocab_file=two)
    same_words = set(vocab_one.keys()) & set(vocab_two.keys())
    different_words_one = set(vocab_one.keys()) - set(vocab_two.keys())
    different_words_two = set(vocab_two.keys()) - set(vocab_one.keys())
    print(f"相同的词有{len(same_words)}个，不同的词有{len(different_words_one)}个和{len(different_words_two)}个")
    with codecs.open(same_file, 'w', encoding='utf-8') as f:
        for i in same_words:
            f.write(f"{i} {vocab_one[i]}\n")
    with codecs.open(diff_one_file, 'w', encoding='utf-8') as f:
        for i in different_words_one:
            f.write(f"{i} {vocab_one[i]}\n")
    with codecs.open(diff_two_file, 'w', encoding='utf-8') as f:
        for i in different_words_two:
            f.write(f"{i} {vocab_two[i]}\n")
    print(f'相同的词保存到{same_file}, 不同的词保存到{diff_one_file}和{diff_two_file}')

def merge_vocab(main_vocab, extra_vocab, output_file="save/merge.vocab"):
    """
    合并两个词表
    :param main_vocab: 主词表
    :param extra_vocab: 额外词表
    :param output_file: 输出文件
    """
    main_vocab = read_vocab_file(main_vocab)
    extra_vocab = read_vocab_file(extra_vocab)
    new_vocab = {**main_vocab, **extra_vocab}
    with codecs.open(output_file, 'w', encoding='utf-8') as f:
        for i, j in new_vocab.items():
            f.write(f"{i} {j}\n")
    print(f"合并词表保存到{output_file}, 共{len(new_vocab)}个词")


if __name__ == '__main__':
    # get_and_save_data()
    my_word_discovery(data_dir="data/tmall", use_cache=False)
    # diff_two_vocab('save/mydata.vocab', 'save/news.vocab')
    # diff_two_vocab('save/wechat.vocab', 'save/news.vocab')
    # 合并same字典里面的词到universal字典里面，作为通用字典
    # merge_vocab(main_vocab="save/universal.vocab", extra_vocab="save/same.vocab", output_file="save/universal.vocab")