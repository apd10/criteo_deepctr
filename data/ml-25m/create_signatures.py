import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import pdb

genome_scores = pd.read_csv("genome-scores.csv")
genome_scores  = genome_scores.pivot_table(values="relevance", index="movieId", columns="tagId").reset_index()
emb = genome_scores.to_numpy()[:, 1:]
for i in range(emb.shape[1]):
    emb[:,i] = (emb[:,i] - np.mean(emb[:,i])) / (0.1 + np.std(emb[:,i]))
srp = np.random.normal(0,1,size=(1128,63))
signs = np.sign(np.matmul(emb, srp))
signatures = (signs + 1)/2
signatures = np.concatenate([np.zeros((signatures.shape[0],1)), signatures], axis=1)
signatures = signatures.astype(np.int64)
x = signatures.reshape(signatures.shape[0], 8, 8)
signatures = np.packbits(x.astype(int), axis=-1).reshape(signatures.shape[0], -1)
signatures = signatures.astype(np.int64)
p = (2**8)**np.arange(8)[::-1]
signatures = signatures * p
signatures = np.sum(signatures, -1)
pdb.set_trace()
fsignatures = np.zeros(206500).astype(np.int64)
fsignatures[genome_scores['movieId'].values] = signatures
np.savetxt("moviesignatures.txt", fsignatures, fmt="%d")
