# -*- coding: utf-8 -*-  # 指定文件编码为 UTF-8

from collections import defaultdict  # 从collections模块导入defaultdict，用于创建字典
import numpy as np  # 导入NumPy库
from rdkit import Chem  # 从RDKit库导入Chem模块，用于化学信息学
import torch  # 导入PyTorch库

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置设备为GPU（如果可用）

# 创建默认字典，用于存储原子、化学键、指纹和边的信息
atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
edge_dict = defaultdict(lambda: len(edge_dict))
radius = 1  # 设置指纹提取的半径


def create_atoms(mol, atom_dict):
    """将分子中的原子类型（例如，H、C和O）转换为索引（例如，H=0，C=1，O=2）。
    注意，每个原子索引考虑了芳香性。
    """
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]  # 获取每个原子的符号
    for a in mol.GetAromaticAtoms():  # 遍历芳香原子
        i = a.GetIdx()  # 获取原子的索引
        atoms[i] = (atoms[i], 'aromatic')  # 将芳香原子标记为芳香
    atoms = [atom_dict[a] for a in atoms]  # 将原子符号转换为索引
    return np.array(atoms)  # 返回原子索引的NumPy数组


def create_ijbonddict(mol, bond_dict):
    """创建一个字典，其中每个键是节点ID，每个值是其邻居节点及化学键（例如，单键和双键）ID的元组。
    """
    i_jbond_dict = defaultdict(lambda: [])  # 创建一个默认字典，用于存储键-值对
    for b in mol.GetBonds():  # 遍历分子的每个化学键
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()  # 获取起始和结束原子的索引
        bond = bond_dict[str(b.GetBondType())]  # 获取化学键类型的索引
        i_jbond_dict[i].append((j, bond))  # 将邻居节点及键类型添加到字典中
        i_jbond_dict[j].append((i, bond))  # 双向添加
    return i_jbond_dict  # 返回原子与其邻居和键的字典


def extract_fingerprints(radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict):
    """基于Weisfeiler-Lehman算法从分子图中提取指纹。
    """
    if (len(atoms) == 1) or (radius == 0):  # 如果只有一个原子或半径为0
        nodes = [fingerprint_dict[a] for a in atoms]  # 直接获取指纹
    else:
        nodes = atoms  # 初始化节点为原子索引
        i_jedge_dict = i_jbond_dict  # 初始化边字典

        for _ in range(radius):  # 循环更新每个节点的指纹
            nodes_ = []  # 初始化新的节点列表
            for i, j_edge in i_jedge_dict.items():  # 遍历每个节点及其边
                neighbors = [(nodes[j], edge) for j, edge in j_edge]  # 获取邻居节点及边信息
                fingerprint = (nodes[i], tuple(sorted(neighbors)))  # 创建新的指纹
                nodes_.append(fingerprint_dict[fingerprint])  # 更新指纹

            # 更新每条边的ID，考虑其两端的节点
            i_jedge_dict_ = defaultdict(lambda: [])  # 初始化新的边字典
            for i, j_edge in i_jedge_dict.items():  # 遍历旧的边字典
                for j, edge in j_edge:  # 遍历每条边
                    both_side = tuple(sorted((nodes[i], nodes[j])))  # 创建排序的节点元组
                    edge = edge_dict[(both_side, edge)]  # 获取边的索引
                    i_jedge_dict_[i].append((j, edge))  # 更新边字典

            nodes = nodes_  # 更新节点
            i_jedge_dict = i_jedge_dict_  # 更新边字典

    return np.array(nodes)  # 返回更新后的指纹数组


def split_dataset(dataset, ratio):
    """打乱并分割数据集。"""
    np.random.seed(1234)  # 固定随机种子以进行可重复性
    # np.random.shuffle(dataset)  # 可选：打乱数据集
    n = int(ratio * len(dataset))  # 根据比例计算分割点
    return dataset[:n], dataset[n:]  # 返回分割后的数据集



def create_testdataset(data_original,device):
    # with open(filepath, 'r') as f:  # 打开文件
    #     # smiles_property = f.readline().strip().split()  # 可选：读取首行
    #     data_original = f.read().strip().split()  # 读取所有数据
    data_original = [data for data in data_original if '.' not in data.split()[0]]  # 过滤掉含有'.'的数据
    dataset = []  # 初始化数据集列表
    for data in data_original:  # 遍历每条数据
        smiles = data  # 读取SMILES字符串
        try:
            """使用上述定义的函数创建每个数据。"""
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))  # 从SMILES生成分子并添加氢
            atoms = create_atoms(mol, atom_dict)  # 创建原子索引
            molecular_size = len(atoms)  # 获取分子大小
            i_jbond_dict = create_ijbonddict(mol, bond_dict)  # 创建原子与键的字典
            fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict)  # 提取指纹
            adjacency = Chem.GetAdjacencyMatrix(mol)  # 获取邻接矩阵
            """将上述每个数据从NumPy转换为在设备（CPU或GPU）上的PyTorch张量。"""
            fingerprints = torch.LongTensor(fingerprints).to(device)  # 转换指纹为张量
            adjacency = torch.FloatTensor(adjacency).to(device)  # 转换邻接矩阵为张量
            proper = torch.LongTensor([int(0)]).to(device)  # 创建标签张量
            dataset.append((smiles, fingerprints, adjacency, molecular_size, proper))  # 添加数据到数据集
        except:
            print(smiles)  # 打印错误的SMILES字符串
    return dataset  # 返回创建的数据集

 
def create_dataset(filename, path, dataname,device):
    dir_dataset = path + dataname  # 构建数据集目录
    print(filename)  # 打印文件名
    """加载数据集。"""
    try:
        with open(dir_dataset + filename, 'r') as f:  # 尝试打开文件
            smiles_property = f.readline().strip().split()  # 读取首行
            data_original = f.read().strip().split('\n')  # 读取所有数据
    except:
        with open(dir_dataset + filename, 'r') as f:  # 如果有异常，重新尝试打开文件
            smiles_property = f.readline().strip().split()  # 读取首行
            data_original = f.read().strip().split('\n')  # 读取所有数据

    # 排除含有 '.' 的数据
    data_original = [data for data in data_original if '.' not in data.split()[0]]
    dataset = []  # 初始化数据集列表
    for data in data_original:  # 遍历每条数据
        # print(data)
        smiles, property = data.strip().split()  # 读取SMILES和属性
        try:
            """使用上述定义的函数创建每个数据。"""
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))  # 从SMILES生成分子并添加氢
            atoms = create_atoms(mol, atom_dict)  # 创建原子索引
            molecular_size = len(atoms)  # 获取分子大小
            i_jbond_dict = create_ijbonddict(mol, bond_dict)  # 创建原子与键的字典
            fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict)  # 提取指纹
            adjacency = Chem.GetAdjacencyMatrix(mol)  # 获取邻接矩阵
            """
            将上述每个数据从NumPy转换为在设备（CPU或GPU）上的PyTorch张量。
            """
            fingerprints = torch.LongTensor(fingerprints).to(device)  # 转换指纹为张量
            adjacency = torch.FloatTensor(adjacency).to(device)  # 转换邻接矩阵为张量
            property = torch.LongTensor([int(property)]).to(device)  # 创建标签张量
            dataset.append((smiles, fingerprints, adjacency, molecular_size, property))  # 添加数据到数据集
        except:
            print(smiles)  # 打印错误的SMILES字符串
    return dataset  # 返回创建的数据集

def get_dcan_dataset(pos_smiles,neg_smiles):#专门为投票模型建立数据集函数
    # try:
    #     with open(dir_dataset + filename, 'r') as f:  # 尝试打开文件
    #         smiles_property = f.readline().strip().split()  # 读取首行
    #         data_original = f.read().strip().split('\n')  # 读取所有数据
    # except:
    #     with open(dir_dataset + filename, 'r') as f:  # 如果有异常，重新尝试打开文件
    #         smiles_property = f.readline().strip().split()  # 读取首行
    #         data_original = f.read().strip().split('\n')  # 读取所有数据

    # 排除含有 '.' 的数据
    pos_data=[i+' 1' for i in pos_smiles]
    neg_data=[i+' 0' for i in neg_smiles]
    data_original=pos_data+neg_data
    data_original = [data for data in data_original if '.' not in data.split()[0]]
    dataset = []  # 初始化数据集列表
    for data in data_original:  # 遍历每条数据
        smiles, property = data.strip().split()  # 读取SMILES和属性
        try:
            """使用上述定义的函数创建每个数据。"""
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))  # 从SMILES生成分子并添加氢
            atoms = create_atoms(mol, atom_dict)  # 创建原子索引
            molecular_size = len(atoms)  # 获取分子大小
            i_jbond_dict = create_ijbonddict(mol, bond_dict)  # 创建原子与键的字典
            fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict)  # 提取指纹
            adjacency = Chem.GetAdjacencyMatrix(mol)  # 获取邻接矩阵
            """
            将上述每个数据从NumPy转换为在设备（CPU或GPU）上的PyTorch张量。
            """
            fingerprints = torch.LongTensor(fingerprints)  # 转换指纹为张量
            adjacency = torch.FloatTensor(adjacency) # 转换邻接矩阵为张量
            property = torch.LongTensor([int(property)])  # 创建标签张量
            dataset.append((smiles, fingerprints, adjacency, molecular_size, property))  # 添加数据到数据集
        except:
            print("DCAN SMILES转化出错")
            print(smiles)  # 打印错误的SMILES字符串
    return dataset  # 返回创建的数据集

def get_dacan_test_dataset(smiles1,device):
    if isinstance(smiles1, str):
        data_original=[smiles1]
        dataset = []  # 初始化数据集列表
        for data in data_original:  # 遍历每条数据
            smiles = data  # 读取SMILES字符串
            try:
                """使用上述定义的函数创建每个数据。"""
                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))  # 从SMILES生成分子并添加氢
                atoms = create_atoms(mol, atom_dict)  # 创建原子索引
                molecular_size = len(atoms)  # 获取分子大小
                i_jbond_dict = create_ijbonddict(mol, bond_dict)  # 创建原子与键的字典
                fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict)  # 提取指纹
                adjacency = Chem.GetAdjacencyMatrix(mol)  # 获取邻接矩阵
                """将上述每个数据从NumPy转换为在设备（CPU或GPU）上的PyTorch张量。"""
                fingerprints = torch.LongTensor(fingerprints).to(device)  # 转换指纹为张量
                adjacency = torch.FloatTensor(adjacency).to(device)  # 转换邻接矩阵为张量
                proper = torch.LongTensor([int(0)]).to(device)  # 创建标签张量
                dataset.append((smiles, fingerprints, adjacency, molecular_size, proper))  # 添加数据到数据集
            except:
                print(smiles)  # 打印错误的SMILES字符串
    return dataset  # 返回创建的数据集

def to_gpu(dataset,device):
    new_data=[]
    for i in dataset:
        new_data.append((i[0],torch.tensor(np.array(i[1])).to(device),torch.tensor(np.array(i[2])).to(device),i[3],torch.tensor(np.array(i[4])).to(device)))
    return new_data