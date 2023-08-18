#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/4 20:28
# @Author  : Feng Zhanpeng
# @File    : 基于Louvain算法识别欺诈团伙.py
# @Software: PyCharm

"""
基于Louvain算法识别欺诈团伙代码执行顺序：
1. 加载Python包；
2. 构建全局社交网络；
3. 基于Louvain算法对全局网络进行社区划分；
4. 基于节点协同分类算法预测非欺诈节点欺诈的概率；
5. 找出疑似欺诈团伙；
"""

# 1. 加载所需Python包

import numpy as np
import pandas as pd
import os
import networkx as nx
import community
import re
import copy
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 2. 构建全局社交网络

# path为network_data.csv和fraud_flag_data.csv的存储路径，在实操时相关路径均要换成自己本地的路径
path = r'D:\\金融风控\\Python金融风控策略实践 代码和数据样例\\Chapter8\\8.3.4基于Louvain算法构建欺诈团伙识别模型\\'
# 建模结果输出路径
path_result = path + 'louvain\\'
if not os.path.exists(path_result):
    os.makedirs(path_result)
# 取数，同时为避免数据重复对数据去重
f = open(path + 'network_data.csv')
mydata = pd.read_csv(f).drop_duplicates()
mydata.describe()

# 基于networkx包对获取到的数据构建全局网络
G = nx.from_pandas_edgelist(mydata, source='source', target='target')

'''
在构建全局网络时，可为节点设置权重，格式如下：
G = nx.from_pandas_edgelist(mydata, source='source',target='target',edge_attr=['weight'])
'''

print('全局网络中节点的数量为:', G.number_of_nodes())
print('全局网络中边的数量为:', G.number_of_edges())

# 计算节点的度数
degrees = pd.DataFrame(columns=['节点和度数'])
degrees['节点和度数'] = pd.Series(G.degree)
degrees['节点'] = degrees['节点和度数'].map(lambda x: x[0])
degrees['度数'] = degrees['节点和度数'].map(lambda x: x[1])
print(degrees.head())

# 计算节点的度中心性
degree_df = pd.DataFrame(columns=['度中心性'])
degree_df['度中心性'] = pd.Series(nx.degree_centrality(G))
degree_df = degree_df.reset_index().rename(columns={'index': 'node'})
print(degree_df.head())

# 计算节点的接近中心性
# closeness_df = pd.DataFrame(columns=['接近中心性'])
# closeness_df['接近中心性'] = pd.Series(nx.closeness_centrality(G))

# 计算节点的中介中心性
# betweenness_df = pd.DataFrame(columns=['中介中心性'])
# betweenness_df['中介中心性'] = pd.Series(nx.betweenness_centrality(G))
# betweenness_df.fillna(0, inplace=True)

# 基于网页排名找出活跃节点
# pagerank_df = pd.DataFrame(columns=['网页排名'])
# pagerank_df['网页排名'] = pd.Series(nx.pagerank(G1))


# 3. 基于Louvain算法对全局网络进行社区划分

# 社区划分，划分结果为字典形式
print('开始基于Louvain算法进行社区划分')
part = community.best_partition(G)
print('社区划分完成,查看划分结果')
print(part)

# 将字典转为数据框
df = pd.DataFrame.from_dict(part, orient='index')
df.reset_index(inplace=True)
# 数据框包含两列，一列表示节点（node），一列表示节点所属的社区(tag)
df.rename(columns={'index': 'node', 0: 'tag'}, inplace=True)
df.head()


# 节点类型解析函数
def match_nodetype(x):
    """ 对节点进行解析，返回每个节点值对应的节点类型
    :param x:  要解析的节点
    :return:   节点类型
    """
    pattern_u = re.compile('^C')
    pattern_p = re.compile('^P')
    pattern_d = re.compile('^D')
    if pattern_u.match(x):
        return '客户节点'
    elif pattern_p.match(x):
        return '电话节点'
    elif pattern_d.match(x):
        return '设备节点'


# 增加节点类型列，为每个节点匹配上对应的节点类型
df['node_type'] = df['node'].map(lambda x: match_nodetype(x))

# 关联节点在全局网络中的度中心
# df=df.merge(degrees,left_on='node',right_on='node',how='left')
# df=df[['tag','node','node_type','node_degree']]


# 4. 基于节点协同分类算法预测非欺诈节点欺诈的概率

# 获取欺诈节点
f = open(path + 'fraud_flag_data.csv')
fraud_flag_data = pd.read_csv(f).drop_duplicates()

# 基于欺诈节点对所有节点进行标注，欺诈为1，否则为0
nodes = list(G.nodes())
node_df = pd.DataFrame({'nodes': nodes})
node_df = pd.merge(node_df, fraud_flag_data, left_on='nodes', right_on='node', how='left')
del node_df['node']
node_df['fraud_flag'] = node_df['fraud_flag'].map(lambda x: 1 if x == 1 else 0)

# 首先获取节点初始欺诈概率，欺诈节点欺诈概率为1，未知点欺诈概率为0
init_fraud_prob = node_df['fraud_flag'].tolist()
fraud_prob = copy.deepcopy(init_fraud_prob)

# 设置节点分类算法迭代次数max_iter，基于六度分离理论，迭代次数通常设置为6就行了
max_iter = 6
# 将每个节点进行编码
nodes_i = {nodes[i]: i for i in range(0, len(nodes))}

# 欺诈节点
fraud_node = node_df['nodes'][node_df['fraud_flag'] == 1].tolist()
# 未知节点
unlabel_nodes = set(nodes).difference(set(fraud_node))

# 开始执行节点协同分类算法，计算未知节点的欺诈概率
for i in range(max_iter):
    print(i)
    pre_fraud_prob = np.copy(fraud_prob)
    for unnode in unlabel_nodes:
        temp = 0
        for item in G.neighbors(unnode):
            temp = temp + pre_fraud_prob[nodes_i[item]]
        fraud_prob[nodes_i[unnode]] = temp / len(list(G.neighbors(unnode)))

# 为每个节点匹配对应的欺诈概率
node_df['fraud_prob'] = [round(i, 3) for i in fraud_prob]

# 基于每个节点的欺诈概率为节点打上是否欺诈的标签，概率值大于0.5为欺诈，否则为非欺诈
node_df['fraud_pred'] = node_df['fraud_prob'].map(lambda x: 1 if x > 0.5 else 0)

# 5. 找出疑似欺诈团伙

# 将社区划分结果与节点欺诈概率预测结果进行关联
df = pd.merge(df, node_df[['nodes', 'fraud_flag', 'fraud_prob', 'fraud_pred']], left_on='node',
              right_on='nodes', how='left')
del df['nodes']

# 为客户、设备、电话节点打上相应的区分标签
df['cust_flag'] = df['node'].map(lambda x: 1 if 'C' in x else 0)
df['device_flag'] = df['node'].map(lambda x: 1 if 'D' in x else 0)
df['phone_flag'] = df['node'].map(lambda x: 1 if 'P' in x else 0)

# 为预测为欺诈的节点打上标签
df['pre_fraud_flag'] = df.apply(lambda x: 1 if (x['fraud_pred'] == 1 and x['fraud_flag'] == 0) else 0, axis=1)

# 计算每个社区对应的疑似欺诈节点占比和节点数
tag_badrate = pd.DataFrame(df.groupby(['tag'])['pre_fraud_flag'].mean().reset_index()).rename(
    columns={'pre_fraud_flag': 'bad_rate'})
tag_num = pd.DataFrame(df['tag'].value_counts().reset_index()).rename(columns={'count': 'tag_num'})
df_01 = pd.merge(df, tag_badrate, left_on='tag', right_on='tag', how='left')
df_02 = pd.merge(df_01, tag_num, left_on='tag', right_on='tag', how='left')

# 基于特定的业务逻辑筛选疑似欺诈社区，具体筛选逻辑可结合实际情况灵活调整。本次筛选节点数大于4且疑似欺诈节点占比大于0的社区认为是疑似欺诈团队
select_tag = df_02[(df_02['tag_num'] > 4) & (df_02['bad_rate'] > 0)]['tag'].unique()
# 打印筛选的社区
print(select_tag)

# 画出疑似欺诈团伙社交网络图
for tag in select_tag:
    tag = int(tag)
    print('疑似欺诈团伙tag为：', tag)
    nodelist = df.loc[df['tag'] == tag, 'node'].tolist()
    print('疑似欺诈团伙节点数为:' + str(len(nodelist)))
    blacklist = df[(df['tag'] == tag) & (df['fraud_flag'] == 1)]['node'].tolist()
    pre_blacklist = df[(df['tag'] == tag) & (df['pre_fraud_flag'] == 1)]['node'].tolist()
    # 团伙中欺诈点占比
    black_rate = '{:.2%}'.format(len(blacklist) / len(nodelist))
    # 团伙中预测为欺诈节点的节点占比
    pre_black_rate = '{:.2%}'.format(len(pre_blacklist) / len(nodelist))
    edgelist = [i for i in G.edges if i[0] in nodelist and i[1] in nodelist]
    # 构建一个无向图
    g = nx.Graph()
    # 将团伙中的节点添加到无向图网络中
    for i in nodelist:
        g.add_node(i)

    for e in edgelist:
        g.add_edge(e[0], e[1])

    pos = nx.spring_layout(g)
    # 基于欺诈团伙节点重构社交网络，并计算网络中节点的度中心性
    subdegree_df = pd.DataFrame(columns=['度中心性'])
    subdegree_df['度中心性'] = pd.Series(nx.degree_centrality(g))
    # 获取度中心最大的top10节点
    top = subdegree_df.sort_values('度中心性', ascending=False)[:10]
    top = top.index.values.tolist()
    # 获取度中心性最大的top10节点中的非欺诈和非预测欺诈的节点
    top = set(top) - set(pre_blacklist) - set(blacklist)
    print(top)

    # 画图时度中心性越大，节点在图中的形状越大
    plt.figure(figsize=(24, 11), dpi=180)
    node_color = [g.degree(v) for v in g]
    node_size = [5000 * nx.degree_centrality(g)[v] for v in g]

    pos = nx.spring_layout(g)
    nx.draw_networkx(g, pos, node_size=node_size, node_color=node_color, alpha=0.8, with_labels=False)

    # 黑节点在图中的字体颜色为黑色
    black_labels = {role: role for role in blacklist}
    nx.draw_networkx_labels(g, pos, labels=black_labels, font_color='k', font_size=16)
    ## 预测的黑节点在图中的字体颜色为红色
    pre_labels = {role: role for role in pre_blacklist}
    nx.draw_networkx_labels(g, pos, labels=pre_labels, font_color='r', font_size=15)

    # 度中心性最大的top10节点中的非欺诈和非预测欺诈的节点在图中的字体颜色为蓝色
    if len(top) > 0:
        top_labels = {role: role for role in top}
        nx.draw_networkx_labels(g, pos, labels=top_labels, font_color='blue', font_size=14)
    plt.title('Tag:' + str(tag) + ',节点数:' + str(
        len(nodelist)) + ',黑节点占比为:' + black_rate + ',预测为坏的节点占比为:' + pre_black_rate, fontsize=20)

    plt.savefig(path_result + 'tag_' + str(tag) + '.png', bbox_inches='tight')
    plt.clf()
    plt.close()
