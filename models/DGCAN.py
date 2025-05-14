# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 18:04:58 2022

@author:Jinyu-Sun
"""

# coding=utf-8
import timeit  # 导入计时模块
import sys  # 导入系统模块
import os
import numpy as np  # 导入numpy库
import math  # 导入数学库
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的功能性神经网络模块
import torch.optim as optim  # 导入PyTorch的优化器模块
import pickle  # 导入pickle模块，用于序列化
from sklearn.metrics import roc_auc_score, roc_curve  # 导入ROC AUC分数和ROC曲线的计算函数
from sklearn.metrics import confusion_matrix  # 导入混淆矩阵的计算函数
# import preprocess as pp  # 导入自定义的预处理模块
import pandas as pd  # 导入Pandas库，用于数据处理
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import dataset.DGCAN_Dataset as pp


# 判断是否可以使用GPU，如果可以则使用GPU
if torch.cuda.is_available():
    device = torch.device('cuda')  # 使用GPU
else:
    device = torch.device('cpu')  # 使用CPU

torch.cuda.empty_cache()  # 清空CUDA缓存


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()  # 初始化父类
        self.dropout = dropout  # dropout率
        self.concat = concat  # 是否拼接
        self.in_features = in_features  # 输入特征的维度
        self.out_features = out_features  # 输出特征的维度
        self.alpha = alpha  # LeakyReLU的负斜率
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))  # 权重矩阵

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))  # 注意力系数
        torch.nn.init.xavier_uniform_(self.W, gain=2.0)  # 使用Xavier初始化权重
        torch.nn.init.xavier_uniform_(self.W, gain=1.9)  # 重新使用Xavier初始化
        self.leakyrelu = nn.LeakyReLU(self.alpha)  # LeakyReLU激活函数

    def forward(self, input, adj):  # 定义前向传播
        """
        input: 输入特征 [N, in_features]，N为节点数量
        adj: 图的邻接矩阵，维度为 [N, N]，非零值为1，表示连接
        """
        # h = torch.mm(input.cpu(), self.W.cpu())  # 计算节点特征的线性变换 [N, out_features]
        h = torch.mm(input, self.W) 
        N = h.size()[0]  # 图中节点的数量
        # 计算注意力系数
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1,
                                                                                          2 * self.out_features)  # [N, N, 2*out_features]
        # e = self.leakyrelu(torch.matmul(a_input.cpu(), self.a.cpu()).squeeze(2))  # 计算注意力权重
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e10 * torch.ones_like(e)  # 创建一个很小的值的向量
        attention = torch.where(adj > 0, e, zero_vec)  # 依据邻接矩阵生成注意力权重
        # 如果邻接矩阵对应位置大于0，则保留该位置的注意力权重；否则设为很小的值
        attention = F.softmax(attention, dim=1)  # 对注意力权重进行softmax归一化
        attention = F.dropout(attention, self.dropout, training=self.training)  # 应用dropout
        h_prime = torch.matmul(attention, h)  # 计算加权特征
        if self.concat:
            return F.elu(h_prime)  # ELU激活后返回
        else:
            return h_prime  # 返回加权特征


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        super(GAT, self).__init__()  # 初始化父类
        """
        n_heads表示有多少个GAT层，最终将这些层拼接在一起，类似于自注意力机制
        用于从不同子空间提取特征。
        """
        self.dropout = dropout  # dropout率
        # 创建多个图注意力层
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # 添加每个注意力层到模型中

        self.out_att = GraphAttentionLayer(nhid, 56, dropout=dropout, alpha=alpha, concat=False)  # 输出的GAT层
        self.nheads = nheads  # 头数

    def forward(self, x, adj):  # 定义前向传播
        x = F.dropout(x, self.dropout, training=self.training)  # 应用dropout
        z = torch.zeros_like(self.attentions[1](x, adj))  # 初始化加权特征
        # 聚合各个注意力层的输出
        for att in self.attentions:
            z = torch.add(z, att(x, adj))  # 累加每个注意力层的输出
        x = z / self.nheads  # 计算各头的平均值
        x = F.dropout(x, self.dropout, training=self.training)  # 再次应用dropout
        x = F.elu(self.out_att(x, adj))  # 经过最后的GAT层和ELU激活
        return F.softmax(x, dim=1)  # 返回经过softmax的输出


class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprints, dim, layer_hidden, layer_output, dropout):
        super(MolecularGraphNeuralNetwork, self).__init__()  # 初始化父类
        self.layer_hidden = layer_hidden  # 隐藏层数量
        self.layer_output = layer_output  # 输出层数量
        self.embed_fingerprint = nn.Embedding(N_fingerprints, dim)  # 定义指纹嵌入层
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_hidden)])  # 定义多个线性层

        self.W_output = nn.ModuleList([nn.Linear(56, 56) for _ in range(layer_output)])  # 定义输出层的线性层
        self.W_property = nn.Linear(56, 2)  # 定义属性预测层

        self.dropout = dropout  # dropout率
        self.alpha = 0.25  # LeakyReLU的负斜率
        self.nheads = 2  # 注意力头数
        self.attentions = GAT(dim, dim, dropout, alpha=self.alpha, nheads=self.nheads).to(device)  # 初始化GAT层并移动到设备

    def pad(self, matrices, pad_value):
        """对矩阵列表进行填充，方便批处理。
        例如，给定矩阵列表 [A, B, C]，
        我们得到新矩阵 [A00, 0B0, 00C]，其中0为填充值。
        """
        shapes = [m.shape for m in matrices]  # 获取矩阵的形状
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])  # 计算填充后的矩阵的大小
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)  # 创建一个全零的填充矩阵
        pad_matrices = pad_value + zeros  # 初始化填充矩阵
        i, j = 0, 0  # 初始化索引
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]  # 获取每个矩阵的形状
            pad_matrices[i:i + m, j:j + n] = matrix  # 将矩阵填充到新的矩阵中
            i += m  # 更新行索引
            j += n  # 更新列索引
        return pad_matrices  # 返回填充后的矩阵

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))  # 计算隐藏向量

        return hidden_vectors + torch.matmul(matrix, hidden_vectors)  # 返回更新后的向量

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]  # 对向量按轴求和
        return torch.stack(sum_vectors)  # 返回堆叠后的向量

    def gnn(self, inputs):
        """将每个输入数据进行拼接或填充以进行批处理。"""
        Smiles, fingerprints, adjacencies, molecular_sizes = inputs  # 解包输入
        fingerprints = torch.cat(fingerprints)  # 拼接指纹
        # fingerprints=fingerprints.cpu()
        adj = self.pad(adjacencies, 0)  # 填充邻接矩阵
        """GNN层（更新指纹向量）。"""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)  # 通过嵌入层生成指纹向量

        for l in range(self.layer_hidden):  # 遍历每个隐藏层
            # hs = self.update(adj.cpu(), fingerprint_vectors.cpu(), l)  # 更新指纹向量
            hs = self.update(adj, fingerprint_vectors, l) 
            fingerprint_vectors = F.normalize(hs, 2, 1)  # 对更新后的向量进行L2归一化
        """注意力层"""
        # molecular_vectors = self.attentions(fingerprint_vectors.cpu(), adj.cpu())  # 通过GAT层获得分子向量
        molecular_vectors = self.attentions(fingerprint_vectors, adj)
        """通过对指纹向量求和或均值获得分子向量。"""
        molecular_vectors = self.sum(molecular_vectors, molecular_sizes)  # 按分子大小进行求和
        return Smiles, molecular_vectors  # 返回SMILES和分子向量

    def mlp(self, vectors):
        """基于多层感知机的回归器。"""
        for l in range(self.layer_output):  # 遍历输出层
            vectors = torch.relu(self.W_output[l](vectors))  # 激活输出向量
        outputs = torch.sigmoid(self.W_property(vectors))  # 经过sigmoid获得最终输出
        return outputs  # 返回输出

    def forward_classifier(self, data_batch, train):
        inputs = data_batch[:-1]  # 获取输入数据
        correct_labels = torch.cat(data_batch[-1])  # 获取正确标签

        if train:  # 如果是训练模式
            Smiles, molecular_vectors = self.gnn(inputs)  # 进行GNN前向传播
            predicted_scores = self.mlp(molecular_vectors)  # 进行MLP前向传播
            '''损失函数'''
            loss = F.cross_entropy(predicted_scores, correct_labels)  # 计算交叉熵损失
            predicted_scores = predicted_scores.to('cpu').data.numpy()  # 将预测分数移到CPU并转换为numpy数组
            predicted_scores = [s[1] for s in predicted_scores]  # 提取第二类的预测分数
            correct_labels = correct_labels.to('cpu').data.numpy()  # 将正确标签移到CPU并转换为numpy数组
            return Smiles, loss, predicted_scores, correct_labels  # 返回SMILES，损失，预测分数和正确标签
        else:  # 如果是测试模式
            with torch.no_grad():  # 不计算梯度
                Smiles, molecular_vectors = self.gnn(inputs)  # 进行GNN前向传播
                predicted_scores = self.mlp(molecular_vectors)  # 进行MLP前向传播
                # loss = F.cross_entropy(predicted_scores.cpu(), correct_labels.cpu())  # 计算交叉熵损失
                loss = F.cross_entropy(predicted_scores, correct_labels)
            predicted_scores = predicted_scores.to('cpu').data.numpy()  # 将预测分数移到CPU并转换为numpy数组
            predicted_scores = [s[1] for s in predicted_scores]  # 提取第二类的预测分数
            correct_labels = correct_labels.to('cpu').data.numpy()  # 将正确标签移到CPU并转换为numpy数组
            
            return Smiles, loss, predicted_scores, correct_labels  # 返回SMILES，损失，预测分数和正确标签
    def test(self, data) :
        self.eval()
        data=pp.create_testdataset(data,device=device)
        DCAN_tester = Tester(self, batch_test=8)
        dcan_prediction_raw = DCAN_tester.test_classifier(data)
        return dcan_prediction_raw


class Trainer(object):
    def __init__(self, model, lr, batch_train):
        self.model = model  # 保存模型
        self.batch_train = batch_train  # 保存训练批次大小
        self.lr = lr  # 保存学习率
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)  # 使用Adam优化器

    def train(self, dataset):
        np.random.shuffle(dataset)  # 打乱数据集
        N = len(dataset)  # 获取数据集的大小
        # N = dataset.shape[0]
        loss_total = 0  # 初始化总损失
        SMILES, P, C = '', [], []  # 初始化SMILES，预测和正确标签列表
        for i in range(0, N, self.batch_train):  # 按批次大小遍历数据集
            data_batch = list(zip(*dataset[i:i + self.batch_train]))  # 获取当前批次数据
            Smiles, loss, predicted_scores, correct_labels = self.model.forward_classifier(data_batch,
                                                                                           train=True)  # 进行前向传播并计算损失
            SMILES += ' '.join(Smiles) + ' '  # 拼接SMILES
            P.append(predicted_scores)  # 保存预测分数
            C.append(correct_labels)  # 保存正确标签
            self.optimizer.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播
            self.optimizer.step()  # 更新参数
            loss_total += loss.item()  # 累加损失
        tru = np.concatenate(C)  # 将所有正确标签合并
        pre = np.concatenate(P)  # 将所有预测分数合并
        AUC = roc_auc_score(tru, pre)  # 计算AUC分数
        SMILES = SMILES.strip().split()  # 去除多余空格并分割SMILES
        pred = [1 if i > 0.15 else 0 for i in pre]  # 根据阈值生成预测标签
        predictions = np.stack((tru, pred, pre))  # 堆叠预测结果
        return AUC, loss_total, predictions  # 返回AUC，损失和预测结果


class Tester(object):
    def __init__(self, model, batch_test):
        self.model = model  # 保存模型
        self.batch_test = batch_test  # 保存测试批次大小

    def test_classifier(self, dataset):
        N = len(dataset)  # 获取数据集的大小
        loss_total = 0  # 初始化总损失
        SMILES, P, C = '', [], []  # 初始化SMILES，预测和正确标签列表
        for i in range(0, N, self.batch_test):  # 按批次大小遍历数据集
            data_batch = list(zip(*dataset[i:i + self.batch_test]))  # 获取当前批次数据
            (Smiles, loss, predicted_scores, correct_labels) = self.model.forward_classifier(data_batch,
                                                                                             train=False)  # 进行前向传播
            SMILES += ' '.join(Smiles) + ' '  # 拼接SMILES
            loss_total += loss.item()  # 累加损失
            P.append(predicted_scores)  # 保存预测分数
            C.append(correct_labels)  # 保存正确标签
        SMILES = SMILES.strip().split()  # 去除多余空格并分割SMILES
        tru = np.concatenate(C)  # 将所有正确标签合并
        pre = np.concatenate(P)  # 将所有预测分数合并
        pred = [1 if i > 0.15 else 0 for i in pre]  # 根据阈值生成预测标签
        # AUC = roc_auc_score(tru, pre)  # 计算AUC分数
        cnf_matrix = confusion_matrix(tru, pred)  # 计算混淆矩阵
        predictions = np.stack((tru, pred, pre))  # 堆叠预测结果
        return  predictions 
        # return predicted_scores

    def save_result(self, result, filename):
        with open(filename, 'a') as f:  # 以追加模式打开文件
            f.write(result + '\n')  # 写入结果

    def save_predictions(self, predictions, filename):
        with open(filename, 'w') as f:  # 以写入模式打开文件
            f.write('Smiles\tCorrect\tPredict\n')  # 写入表头
            f.write(predictions + '\n')  # 写入预测结果

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)  # 保存模型参数


def dump_dictionary(dictionary, filename):
    with open('../DGCAN/model' + filename, 'wb') as f:  # 以二进制写入模式打开文件
        pickle.dump(dict(dictionary), f)  # 将字典写入文件



class Tester(object):
    def __init__(self, model, batch_test):
        self.model = model  # 保存模型
        self.batch_test = batch_test  # 保存测试批次大小

    def test_classifier(self, dataset):
        N = len(dataset)  # 获取数据集的大小
        loss_total = 0  # 初始化总损失
        SMILES, P, C = '', [], []  # 初始化SMILES，预测和正确标签列表
        for i in range(0, N, self.batch_test):  # 按批次大小遍历数据集
            data_batch = list(zip(*dataset[i:i + self.batch_test]))  # 获取当前批次数据
            (Smiles, loss, predicted_scores, correct_labels) = self.model.forward_classifier(data_batch,
                                                                                             train=False)  # 进行前向传播
            SMILES += ' '.join(Smiles) + ' '  # 拼接SMILES
            loss_total += loss.item()  # 累加损失
            P.append(predicted_scores)  # 保存预测分数
            C.append(correct_labels)  # 保存正确标签
        SMILES = SMILES.strip().split()  # 去除多余空格并分割SMILES
        tru = np.concatenate(C)  # 将所有正确标签合并
        pre = np.concatenate(P)  # 将所有预测分数合并
        pred = [1 if i > 0.15 else 0 for i in pre]  # 根据阈值生成预测标签
        # AUC = roc_auc_score(tru, pre)  # 计算AUC分数
        cnf_matrix = confusion_matrix(tru, pred)  # 计算混淆矩阵
        predictions = np.stack((tru, pred, pre))  # 堆叠预测结果
        return  predictions[2,:]
        # return predicted_scores
