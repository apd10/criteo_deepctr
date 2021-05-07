# -*- coding: utf-8 -*-
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *
from torch import nn
import numpy as np
import argparse
import yaml
import json
import pdb
import sys

from torch.utils.tensorboard import SummaryWriter

def count_parameters(model):
    total_para = 0
    strr = "PARAMETERS:\n"
    for name,parameters in model.named_parameters():
        strr += name + ':' + str(parameters.size()) + "\n"
        total_para += parameters.numel()
    strr += "Total:" + str(total_para) + "\n"
    return strr

def load_data_in_df(args, config):
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    target = ['label']

    if args.test_run:
        print("SAMPLE RUN...")
        #df =  pd.read_csv('/home/apd10/DeepCTR-Torch/examples/criteo_sample.txt')
        df =  pd.read_csv('/home/apd10/dlrm/dlrm/input/small_data.csv')

        df[sparse_features] = df[sparse_features].fillna('-1', )
        df[dense_features] = df[dense_features].fillna(0, )

        # 1.Label Encoding for sparse features,and do simple Transformation for dense features
        for feat in sparse_features:
            lbe = LabelEncoder()
            df[feat] = lbe.fit_transform(df[feat])
        mms = MinMaxScaler(feature_range=(0, 1))
        df[dense_features] = mms.fit_transform(df[dense_features])
    else:
        handle = np.load("/home/apd10/dlrm/dlrm/input/data.npz")
        if "data" in config:
            if config["data"]["type"] == "le2":
                print("Using le2 data")
                handle = np.load("/home/apd10/dlrm/dlrm/input/data.le2.npz")
            

        intdf = pd.DataFrame(handle['intF'], columns = dense_features)
        catdf = pd.DataFrame(handle['catF'], columns = sparse_features)
        labeldf = pd.DataFrame(handle['label'], columns = target)
        df  = pd.concat([labeldf, intdf, catdf], axis=1)
        del intdf, catdf, labeldf
        # all the above processing like label encoding, mms etc is already done in dumped data.npz
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action="store", dest="config", type=str, default=None, required=True,
                    help="config to setup the training")
    parser.add_argument('--test_run', action="store_true", default=False)
    parser.add_argument('--epochs', action="store", dest="epochs", default=30)

    args = parser.parse_args()
    config_file = args.config
    with open(config_file, "r") as f:
        config = yaml.load(f)

    summaryWriter = SummaryWriter()
    commandlinefile = summaryWriter.log_dir + "/cmd_args.txt" 
    with open(commandlinefile, "w") as f:
       f.write(json.dumps(vars(args)))

    out_handle = open(config["results"] +".log", "a")
    if not args.test_run:
        sys.stdout = out_handle
    else:
        out_handle = sys.stdout
    

    data = load_data_in_df(args, config)

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    target = ['label']

    # 2.count #unique features for each sparse field,and record dense feature field name
    embedding_params = config["embedding"]
    embedding_dim = embedding_params["size"]
    if embedding_params["etype"] == "full":
        fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim)
                                  for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                                  for feat in dense_features]
    elif embedding_params["etype"] == "rma":
        print("FIGURE OUT THE INITIALIZATION")
        hashed_weight = nn.Parameter(torch.from_numpy(np.random.uniform(
                        low=-np.sqrt(1 / embedding_dim), high=np.sqrt(1 / embedding_dim), size=((embedding_params["rma"]["memory"],))
                ).astype(np.float32)))
        fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim, use_rma=True, hashed_weight=hashed_weight)
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]

    elif embedding_params["etype"] == "lma":
        print("FIGURE OUT THE INITIALIZATION")
        lma_params = embedding_params["lma"]
        hashed_weight = nn.Parameter(torch.from_numpy(np.random.uniform(
                        low=-np.sqrt(1 / embedding_dim), high=np.sqrt(1 / embedding_dim), size=((lma_params["memory"],))
                ).astype(np.float32)))
        signature = np.load(lma_params["signature"])["signature"]
        signature = torch.from_numpy(signature).to("cuda:0")
        fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim, use_lma=True, hashed_weight=hashed_weight, 
                                              key_bits=lma_params["key_bits"], keys_to_use=lma_params["keys_to_use"], signature=signature)
                                    for feat in sparse_features]  + [DenseFeat(feat, 1, ) for feat in dense_features]
    elif embedding_params["etype"] == "rma_bern":

        rma_bern_params = embedding_params["rma_bern"]

        ls = [ int(x) for x in rma_bern_params["mlp"].split('-')]
        mlp_model = nn.ModuleList()
        for i in range(0, len(ls) - 2):
            mlp_model.append(nn.Linear(ls[i], ls[i+1]))
            mlp_model.append(nn.ReLU())
        mlp_model.append(nn.Linear(ls[len(ls)-2], ls[len(ls) - 1]))
        bern_mlp_model = torch.nn.Sequential(*mlp_model).to("cuda:0")

        fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), int(ls[-1]), use_rma_bern=True, bern_embedding_dim=ls[0], bern_mlp_model=bern_mlp_model)
                                    for feat in sparse_features]  + [DenseFeat(feat, 1, ) for feat in dense_features]
    else:
        raise NotImplementedError
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.1, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'


    #initialize model
    if config["model"] == "deepfm":
          params = config["deepfm"]
          model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                    dnn_hidden_units=(400,400,400),
                    dnn_dropout=0.5,
                    task='binary',
                    l2_reg_embedding=params["reg"], l2_reg_linear=params["reg"], device=device)
    else:
        raise NotImplementedError


    print(model)
    strr = count_parameters(model)
    out_handle.write(strr)
    out_handle.flush()

    model.compile("adam", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )

    history = model.fit(train_model_input, train[target].values, batch_size=2048, epochs=int(args.epochs), verbose=2,
                        validation_split=0.2, summaryWriter=summaryWriter)
    pd.DataFrame(history.history).to_csv(config["results"]+".results.csv")
    pred_ans = model.predict(test_model_input, 16348)
    out_handle.write("test LogLoss: "+str( round(log_loss(test[target].values, pred_ans), 4))+"\n")
    out_handle.write("test AUC"+str( round(roc_auc_score(test[target].values, pred_ans), 4))+"\n")
    out_handle.close()
