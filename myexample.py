#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2022/8/17 16:22
# @File  : myexample.py
# @Author: 
# @Desc  : 利用自己数据测试新词发现功能
import os
import glob
import re
import codecs
import pandas as pd
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
    txts = glob.glob(f'{data_dir}/jd/*.csv')
    assert len(txts) > 0, "没有找到数据文件, 请检查文件"
    for txt in txts:
        d = codecs.open(txt, encoding='utf-8').read()
        d = d.replace(u'\u3000', ' ').strip()
        data = re.sub(u'[^\u4e00-\u9fa50-9a-zA-Z ]+', '\n', d)
        if use_lines:
            for line in data.split('\n'):
                yield line
        else:
            yield data
def my_data_discovery(save_dir="save", use_cache=False):
    min_count = 32
    order = 4
    corpus_file = os.path.join(save_dir, 'mydata.corpus')  # 语料保存的文件名
    vocab_file = os.path.join(save_dir, 'mydata.chars')  # 字符集保存的文件名
    ngram_file = os.path.join(save_dir, 'mydata.ngrams')  # ngram集保存的文件名
    output_file = os.path.join(save_dir, 'mydata.vocab')  # 最后导出的词表文件名
    memory = 0.5  # memory是占用内存比例，理论上不能超过可用内存比例
    print(f"开始构建语料库，保存至{corpus_file}")
    # write_corpus(text_generator(), corpus_file)  # 将语料转存为文本
    print(f"开始计算ngram，保存至{ngram_file}")
    # count_ngrams(corpus_file, order, vocab_file, ngram_file, memory)  # 用Kenlm统计ngram
    print(f"加载由kelm构建好的ngram")
    ngrams = KenlmNgrams(vocab_file, ngram_file, order, min_count)  # 加载ngram
    print(f"互信息过滤ngrams")
    output_ngrams = filter_ngrams(ngrams.ngrams, ngrams.total, [0, 2, 4, 6])  # 过滤ngram
    ngtrie = SimpleTrie()  # 构建ngram的Trie树
    print(f"构建ngram的Trie树")
    for w in Progress(output_ngrams, 10000, desc=u'构建 ngram trie'):
        # w 是每个词， eg: 'B5修复'
        ngtrie.add_word(w)
    print(f"最终ngram trie 构建的字典数量是: {len(ngtrie.dic)} 个")
    print(f"开始发现新词")
    candidates = {}  # 得到候选词
    for t in Progress(text_generator(use_lines=True), 2, desc='发现新词中'):
        if len(candidates) % 1000:
            print(f"已经发现{len(candidates)}个候选新词")
        for w in ngtrie.tokenize(t):  # 预分词
            candidates[w] = candidates.get(w, 0) + 1

    # 频数过滤
    candidates = {i: j for i, j in candidates.items() if j >= min_count}
    # 互信息过滤(回溯)
    print(f"互信息过滤词表")
    candidates = filter_vocab(candidates, ngrams, order)

    # 输出结果文件
    with codecs.open(output_file, 'w', encoding='utf-8') as f:
        for i, j in sorted(candidates.items(), key=lambda s: -s[1]):
            s = '%s %s\n' % (i, j)
            f.write(s)

if __name__ == '__main__':
    # get_and_save_data()
    my_data_discovery()