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
import shutil

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
    x = np.load(args.data)
    train_data = x['train']
    test_data = x['test']

    ltrain = train_data.shape[0]
    ltest = test_data.shape[0]

    data = np.concatenate([train_data, test_data], axis=0)

    columns = np.loadtxt(args.data_columns, dtype=str)
    df = pd.DataFrame(data, columns=columns)

    del df['timestamp']

    sparse_features = ['userId', 'movieId']
    target = ['rating']
    dense_features = [c for c in df.columns if c not in sparse_features + target]
    print(len(dense_features), len(target), len(sparse_features), len(df.columns))

    df[sparse_features] = df[sparse_features].fillna('-1', )
    df[dense_features] = df[dense_features].fillna(0, )
    
    for f in sparse_features + target:
        df[f] = df[f].astype(int)


    for feat in sparse_features:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])

    mms = MinMaxScaler(feature_range=(0, 1))
    df[dense_features] = mms.fit_transform(df[dense_features])
    
    intdf = df[dense_features]
    catdf = df[sparse_features]
    labeldf = df[target[0]]
    df  = pd.concat([labeldf, intdf, catdf], axis=1)
    del intdf, catdf, labeldf
    
    return df, ltrain, ltest

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action="store", dest="config", type=str, default=None, required=True,
                    help="config to setup the training")
    parser.add_argument('--test_run', action="store_true", default=False)
    parser.add_argument('--epochs', action="store", dest="epochs", default=25, type=int)
    parser.add_argument('--data', action="store", dest="data", default="/home/apd10/criteo_deepctr/data/ml-25m/data-small.npz", type=str)
    parser.add_argument('--data_columns', action="store", dest="data_columns", default="/home/apd10/criteo_deepctr/data/ml-25m/columns.txt", type=str)
    

    args = parser.parse_args()
    config_file = args.config
    with open(config_file, "r") as f:
        config = yaml.load(f)
    print("config", config)

    summaryWriter = SummaryWriter()
    commandlinefile = summaryWriter.log_dir + "/cmd_args.txt" 
    configfile = summaryWriter.log_dir + "/config.yml" 
    shutil.copyfile(config_file, configfile)
    with open(commandlinefile, "w") as f:
       f.write(json.dumps(vars(args)))
    

    out_handle = open(summaryWriter.log_dir +"/res.log", "a")
    if not args.test_run:
        sys.stdout = out_handle
    else:
        out_handle = sys.stdout
    

    data,ltrain,ltest = load_data_in_df(args, config)
    train_df, test_df = data.head(ltrain), data.tail(ltest)
    
    movieIdCount = pd.value_counts(train_df['movieId'])
    movie_id_count_df = pd.DataFrame({'movieId' : movieIdCount.index, 'count' : movieIdCount.values, 'rank' : np.arange(len(movieIdCount))})   

    sparse_features = ['userId', 'movieId']
    target = ['rating']
    dense_features = [c for c in data.columns if c not in sparse_features + target]

    # 2.count #unique features for each sparse field,and record dense feature field name
    embedding_params = config["embedding"]
    embedding_dim = embedding_params["size"]
    if embedding_params["etype"] == "full":
        fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim)
                                  for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                                  for feat in dense_features]
    elif embedding_params["etype"] == "remb_rma":
        print ("FIGURE OUT THE INITIALIZATION")
        sfeats = []
        for feat in sparse_features:
            if feat == "movieId":
                pms = embedding_params["remb_rma"][feat]
                sfeats.append(SparseFeat(feat, data[feat].nunique(), embedding_dim, 
                                    remb=True, mode=pms["mode"], n=pms["n"], k=pms["k"], d=pms["d"], best=pms["best"], sorted_vocab=torch.from_numpy(np.array(movieIdCount.index).reshape(-1)), full_size = pms["full_size"], signature=None))
            else:
                sfeats.append(SparseFeat(feat, data[feat].nunique(), embedding_dim))

        fixlen_feature_columns = sfeats  + [DenseFeat(feat, 1,) for feat in dense_features]

    elif embedding_params["etype"] == "remb_mh":
        print ("FIGURE OUT THE INITIALIZATION")
        sfeats = []
        for feat in sparse_features:
            if feat == "movieId":
                pms = embedding_params["remb_mh"][feat]
                signatures = np.load(pms["minhashes"])["hashes"]
                signatures = torch.from_numpy(signatures).long().to("cuda:0")
                sfeats.append(SparseFeat(feat, data[feat].nunique(), embedding_dim, 
                                    remb=True, mode=pms["mode"], n=pms["n"], k=pms["k"], d=pms["d"], best=pms["best"], sorted_vocab=torch.from_numpy(np.array(movieIdCount.index).reshape(-1)), full_size = pms["full_size"], signatures=signatures))
            else:
                sfeats.append(SparseFeat(feat, data[feat].nunique(), embedding_dim))

        fixlen_feature_columns = sfeats  + [DenseFeat(feat, 1,) for feat in dense_features]

    elif embedding_params["etype"] == "rma":
        print("FIGURE OUT THE INITIALIZATION")
        sfeats = []
        for feat in sparse_features:
            if feat == "movieId":
                hashed_weight = nn.Parameter(torch.from_numpy(np.random.uniform(
                          low=-0.001, high=0.001, size=(int(embedding_params["rma"]["compression"] * embedding_dim * data[feat].nunique()),)
                  ).astype(np.float32)))
                sfeats.append(SparseFeat(feat, data[feat].nunique(), embedding_dim, use_rma=True, hashed_weight=hashed_weight))
            else:
                sfeats.append(SparseFeat(feat, data[feat].nunique(), embedding_dim))

        fixlen_feature_columns = sfeats  + [DenseFeat(feat, 1,)for feat in dense_features]

    elif embedding_params["etype"] == "lma":
        print("FIGURE OUT THE INITIALIZATION")
        sfeats = []
        for feat in sparse_features:
            if feat == "movieId":
                print("lma for movieId")
                lma_params = embedding_params["lma"]
                hashed_weight = nn.Parameter(torch.from_numpy(np.random.uniform(
                            low=-0.004, high=0.004, size=((lma_params["memory"],))
                    ).astype(np.float32)))
                signature = np.loadtxt(lma_params["signature"], dtype=np.int64)
                signature = torch.from_numpy(signature).to("cuda:0")
                sfeats.append(SparseFeat(feat, data[feat].nunique(), embedding_dim, use_lma=True, hashed_weight=hashed_weight, 
                                              key_bits=lma_params["key_bits"], keys_to_use=lma_params["keys_to_use"], signature=signature))
            else:
                sfeats.append(SparseFeat(feat, data[feat].nunique(), embedding_dim))


        fixlen_feature_columns = sfeats  + [DenseFeat(feat, 1,)for feat in dense_features]
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

    # Question(yanzhou): why dnn_feature_columns and linear_feature_columns are the same?
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train_model_input = {name: train_df[name] for name in feature_names}
    test_model_input = {name: test_df[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    if "seed" in config:
        model_seed = config["seed"]
    else:
        model_seed = 1024 # default seed

    #initialize model
    if config["model"] == "deepfm":
          params = config["deepfm"]
          model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                    dnn_hidden_units=(400,400,400),
                    dnn_dropout=0.5,
                    task='regression',
                    l2_reg_embedding=0, l2_reg_linear=0, device=device, seed=model_seed)
    elif config["model"] == "dcn":
        params = config["dcn"]
        model = DCN(linear_feature_columns=linear_feature_columns, 
        dnn_feature_columns=dnn_feature_columns, 
        dnn_hidden_units=(1024,1024,1024,1024),
        cross_num = 1,
        l2_reg_linear = 0, l2_reg_embedding=0, l2_reg_cross =0, l2_reg_dnn=0,
        device=device, seed=model_seed)
    elif config["model"] == "onn":
        params = config["onn"]
        model = ONN(linear_feature_columns=linear_feature_columns, 
        dnn_feature_columns=dnn_feature_columns, 
        dnn_hidden_units=(400,400,400),
        l2_reg_linear = 0, l2_reg_embedding=0, l2_reg_dnn=0,
        device=device, seed=model_seed)
    elif config["model"] =="fibinet":
        params = config["fibinet"]
        model = FiBiNET(linear_feature_columns=linear_feature_columns, 
        dnn_feature_columns=dnn_feature_columns, 
        dnn_hidden_units=(400,400,400),
        dnn_dropout=0.5,
        device=device, seed=model_seed)

    elif config["model"] =="xdeepfm":
        params = config["xdeepfm"]
        model = xDeepFM(linear_feature_columns=linear_feature_columns, 
        dnn_feature_columns=dnn_feature_columns,
        dnn_hidden_units=(400,400,400),
        cin_layer_size=(200,200,200),
        dnn_dropout=0.5,
        l2_reg_linear = 0.0001, l2_reg_embedding=0.0001, l2_reg_dnn=0.0001,
        device=device, seed=model_seed)
  
    elif config["model"] =="autoint":
        params = config["autoint"]
        model = AutoInt(linear_feature_columns=linear_feature_columns, 
        dnn_feature_columns=dnn_feature_columns, att_embedding_size=32, dnn_hidden_units=(400,400,400), l2_reg_dnn=0, l2_reg_embedding=0,
        device=device, seed=model_seed)
    else:
        raise NotImplementedError


    print(model)
    strr = count_parameters(model)
    out_handle.write(strr)
    out_handle.flush()

    model.compile("adam", "mse", metrics=['mse'], )

    min_test_iteration = 25000 # end of 1 epoch
    batch_size = config["train"]["batch_size"]

    if args.test_run:
        min_test_iteration = -1
        batch_size = 128

    history = model.fit(train_model_input, train_df[target].values, batch_size=batch_size, epochs=args.epochs, verbose=2,
                        validation_split=0.2, summaryWriter=summaryWriter, test_x = test_model_input, test_y = test_df[target].values,
                        min_test_iteration = min_test_iteration, eval_every=10000, block_eval_helper=movie_id_count_df)
    #pd.DataFrame(history.history).to_csv(summaryWriter.log_dir + "/results.csv", index=False)
    out_handle.close()
