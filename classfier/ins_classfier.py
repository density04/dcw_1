import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
from models import RNNLM, GCNModel, DGCAN  #import models相当于执行了__init__.py的内容,这里相当于从__init__.py里面引用RNNLM
import numpy as np
import dataset.DGCAN_Dataset as pp
import pandas as pd


def ins_classifer(filepath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pos_data=list(open("data/ins1.txt"))
    neg_data=list(open("data/neg.txt"))[:len(pos_data)]
    dcan_data=pp.get_dcan_dataset(pos_data,neg_data) #设置原子字典

    data=list(open(filepath))

    gcn_model = GCNModel(input_size=31, hidden_size1=32, hidden_size2=256, n_layers=4, n_heads=4, dropout=0.2)
    gcn_model.load_state_dict(torch.load('para/gcn_model_all_ins.pth'))
    gcn_model=gcn_model.to(device)
    gcn_prediction_raw=gcn_model.test(data)

    rnn_model = RNNLM(input_size=66,stereo=True,hidden_size=1024,n_layers=4,dropout=0.2)
    rnn_model.load_state_dict(torch.load('para/rnn_model_all_ins.pth'))
    rnn_model=rnn_model.to(device)
    rnn_prediction_raw=rnn_model.test(data)

    DCAN_model = DGCAN.MolecularGraphNeuralNetwork(5000,dim=52, layer_hidden=4, layer_output=10, dropout=0.45)
    DCAN_model.load_state_dict(torch.load('para/DCAN_model_all_ins.pth'))
    DCAN_model=DCAN_model.to(device)
    dcan_prediction_raw=DCAN_model.test(data=data)
    
    
    
    result=pd.DataFrame({'smiles':data,'rnn':np.zeros(len(data)),'gcn':np.zeros(len(data)),'dcan':np.zeros(len(data))})
    result['rnn']=rnn_prediction_raw
    result['gcn']=gcn_prediction_raw
    result['dcan']=dcan_prediction_raw[2,:]
    return result

if __name__=='__main__':
    filepath="data/1.txt"
    res=ins_classifer(filepath)
    print(res)