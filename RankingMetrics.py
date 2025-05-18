#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
precision_recall_ndcg_at_k和map_mrr_ndcg为推荐评估指标

1.precision_recall分为：
正确率 = 提取出的正确信息条数 / 提取出的信息条数    
召回率 = 提取出的正确信息条数 / 样本中的信息条数 
两者取值在0和1之间，数值越接近1，查准率或查全率就越高。
F1值  = 正确率 * 召回率 * 2 / (正确率 + 召回率) （F值即为正确率和召回率的调和平均值）。
2.MAP
主集合的平均准确率(MAP)是每个主题的平均准确率的平均值。
MAP 是反映系统在全部相关文档上性能的单值指标。
系统检索出来的相关文档越靠前(rank 越高)，MAP就可能越高。
如果系统没有返回相关文档，则准确率默认为0。
3.MRR
是把标准答案在被评价系统给出结果中的排序取倒数作为它的准确度
再对所有的问题取平均。
4.NDCG
NDCG中，相关度分成从0到r的r+1的等级(r可设定)
一般情况下用户会优先点选排在前面的搜索结果，
所以应该引入一个折算因子(discounting factor): 
    log(2)/log(1+rank)
DCG = Gain * log(3)/log(1+rank)
NDCG = GCG/Max(DCG)

https://www.cnblogs.com/baiting/p/5138757.html
'''
import math
import numpy as np

def precision_recall_ndcg_at_k(k, rankedlist, test_matrix):
    idcg_k = 0
    dcg_k = 0
    n_k = k if len(test_matrix) > k else len(test_matrix)#nk=1
    for i in range(n_k):
        idcg_k += 1 / math.log(i + 2, 2)
    
    b1 = rankedlist
    b2 = test_matrix
    s2 = set(b2)
    hits = [(idx, val) for idx, val in enumerate(b1) if val in s2]
    
    count = len(hits)
    
    for c in range(count):
        dcg_k += 1 / math.log(hits[c][0] + 2, 2) #1/log2(3)

    return float(count / k), float(count / len(test_matrix)), float(dcg_k / idcg_k)

def map_mrr_ndcg(rankedlist, test_matrix):
    ap = 0
    map = 0
    dcg = 0
    idcg = 0
    mrr = 0
    for i in range(len(test_matrix)):
        idcg += 1 / math.log(i + 2, 2)

    b1 = rankedlist
    b2 = test_matrix
    s2 = set(b2)
    hits = [(idx, val) for idx, val in enumerate(b1) if val in s2]
    count = len(hits)

    for c in range(count):
        ap += (c + 1) / (hits[c][0] + 1)
        dcg += 1 / math.log(hits[c][0] + 2, 2)

    if count != 0:
        mrr = 1 / (hits[0][0] + 1)

    if count != 0:
        map = ap / count

    return map, mrr, float(dcg / idcg)




