import pandas as pd
import numpy as np
from sklearn.utils import shuffle


#ratings = pd.read_csv("ratings.csv")
#movies = pd.read_csv("movies.csv")
#genome_scores = pd.read_csv("genome-scores.csv")
#genome_scores  = genome_scores.pivot_table(values="relevance", index="movieId", columns="tagId").reset_index()
tags = pd.read_csv("tags.csv")
genome_tags = pd.read_csv("genome_tags.csv")
genome_tags = pd.read_csv("genome-tags.csv")
tags
genome_tags
import pandas as pd
import numpy as np
from sklearn.utils import shuffle


#ratings = pd.read_csv("ratings.csv")
#movies = pd.read_csv("movies.csv")
genome_scores = pd.read_csv("genome-scores.csv")
genome_scores  = genome_scores.pivot_table(values="relevance", index="movieId", columns="tagId").reset_index()
genome_scores
genome_scores['movieId'].unique()
movies = genome_scores['movieId'].unique()
tags
tags['movieId']
tags['movieId'].unique()
movies2 = tags['movieId'].unique()
movies1
movies
len(movies)
len(movies2)
np.concatenate([movies, movies2])
np.concatenate([movies, movies2]).unique()
pd.Series(np.concatenate([movies, movies2])).unique()
x = pd.Series(np.concatenate([movies, movies2])).unique()
len(x)
tags
tags
gb = tags.groupby("UserId")
gb = tags.groupby("movieId")
gv
gb['tag'].transform(lambda x: ','.join(x))
gb
tags
gb['tag'].apply(','.join).reset_index()
less tags
tags.iloc[162]
tags.iloc[161]
tags.iloc[162]
tags.iloc[163]
gb = tags.groupby("movieId")
gb['tag']
gb = tags.head(100).groupby("movieId")
gb['tag'].apply(','.join).reset_index()
gb = tags.groupby("movieId")
gb['tag']
gb['tag'].iloc[162]
gb['tag'].apply(','.join).reset_index()
gb['tag'].first().reset_index()
gb['tag'].first().reset_index().iloc[162]
tags[tags['tag'] == 'gay']
tags[tags['movieId'] == 171]
tags['tag']
tags['tag'].str
tags['tag'].str()
tags['tag'] = tags['tag'].astype(str)
gb = tags.groupby("movieId")
gb['tag'].first().reset_index()
gb['tag'].apply(','.join).reset_index()
gb['tag'].unique().apply(','.join).reset_index()
tags = gb['tag'].unique().apply(','.join).reset_index()
tags
tags['tag'][0]
tags['tag'][1]
tags['tag'][45250]
tags.to_csv("temp.csv")
history
import pandas as pd
tags = pd.read_csv("temp.csv")
ls
tags
def get_tokens(s):
    tokens = []
    for big_token in s.split(','):
        for i in range(len(big_token) - 5):
            tokens.append(big_token[i:i+5])
tags['tag'][0]
get_tokens(tags['tag'][0])
def get_tokens(s):
    tokens = []
    for big_token in s.split(','):
        for i in range(len(big_token) - 5):
            tokens.append(big_token[i:i+5])
    return tokens
get_tokens(tags['tag'][0])
def get_tokens(s):
    tokens = []
    for big_token in s.split(','):
        for token in big_token.split(' :')
            tokens.append(token.lower())
    return tokens
def get_tokens(s):
    tokens = []
    for big_token in s.split(','):
        for token in big_token.split(' :'):
            tokens.append(token.lower())
    return tokens
get_tokens(tags['tag'][0])
def get_tokens(s):
    tokens = []
    for big_token in s.split(','):
        for token in big_token.split(' '):
            tokens.append(token.lower())
    return tokens
get_tokens(tags['tag'][0])
def get_tokens(s):
    tokens = []
    for big_token in s.split(','):
        for token in big_token.split(' '):
            tokens.append(token.lower())
    return list(set(tokens))
get_tokens(tags['tag'][0])
get_tokens(tags['tag'][1])
import re
def get_tokens(s):
    tokens = []
    for token in re.split('[, :)(]', s):
        tokens.append(token.lower())
    return list(set(tokens))
get_tokens(tags['tag'][1])
get_tokens(tags['tag'][0])
get_tokens(tags['tag'][100])
get_tokens(tags['tag'][101])
get_tokens(tags['tag'][123])
get_tokens(tags['tag'][1000])
get_tokens(tags['tag'][1001])
get_tokens(tags['tag'][10012])
import re
def get_tokens(s):
    tokens = []
    for token in re.split('[, :)(]', s):
        tokens.append(token.lower()[:5])
    return list(set(tokens))
get_tokens(tags['tag'][10012])
import re
def get_tokens(s):
    tokens = []
    for token in re.split('[, :)(]', s):
        tokens.append(token.lower()[:6])
    return list(set(tokens))
get_tokens(tags['tag'][10012])
import mmh3
pip install murmurhash3
import mmh3
mmh3
import re
def get_tokens(s):
    tokens = []
    for token in re.split('[, :)(]', s):
        tokens.append(mmh3(token.lower()[:6]))
    return list(set(tokens))
import re
def get_tokens(s):
    tokens = []
    for token in re.split('[, :)(]', s):
        tokens.append(mmh3(token.lower()[:6], 41212))
    return list(set(tokens))
get_tokens(tags['tag'][10012])
import re
def get_tokens(s):
    tokens = []
    for token in re.split('[, :)(]', s):
        tokens.append(mmh3.hash(token.lower()[:6], 41212))
    return list(set(tokens))
get_tokens(tags['tag'][10012])
mmh3.hash?
import re
def get_tokens(s):
    tokens = []
    for token in re.split('[, :)(]', s):
        tokens.append(abs(mmh3.hash(token.lower()[:6], 41212)))
    return list(set(tokens))
get_tokens(tags['tag'][10012])
import re
def get_tokens(s):
    tokens = []
    for token in re.split('[, :)(]', s):
        tokens.append(abs(mmh3.hash(token.lower()[:6], 41212)))
    tokens = list(set(tokens))
import re
def get_tokens(s):
    tokens = []
    for token in re.split('[, :)(]', s):
        tokens.append(abs(mmh3.hash(token.lower()[:6], 41212)))
    tokens = set(tokens)
    hashes = []
    for i in range(10):
        l = [abs(mmh3.hash(str(t), 232323+i)) for t in tokens]
        hashes.append(np.min(l))
   return hashes
import re
def get_tokens(s):
    tokens = []
    for token in re.split('[, :)(]', s):
        tokens.append(abs(mmh3.hash(token.lower()[:6], 41212)))
    tokens = set(tokens)
    hashes = []
    for i in range(10):
        l = [abs(mmh3.hash(str(t), 232323+i)) for t in tokens]
        hashes.append(np.min(l))
    return hashes
get_tokens(tags['tag'][10012])
import pandas as pd
import numpy as np
get_tokens(tags['tag'][10012])
get_tokens(tags['tag'][111])
get_tokens(tags['tag'][1123])
get_tokens(tags['tag'][1123]) % 1000000
np.array(get_tokens(tags['tag'][1123])) % 1000000
np.array(get_tokens(tags['tag'][1])) % 1000000
np.array(get_tokens(tags['tag'][12])) % 1000000
np.array(get_tokens(tags['tag'][132])) % 1000000
tags
hashes = np.zeros(209063, 10)
hashes = np.zeros((209063, 10))
hashes
for id in range(hashes.shape[0]):
    tag = tags[tags['movieId'] == id]
    print(tag)
for id in range(hashes.shape[0]):
    tag = tags[tags['movieId'] == id]
    print(tag)
    break
import re
def get_tokens(s):
    tokens = []
    for token in re.split('[, :)(]', s):
        tokens.append(abs(mmh3.hash(token.lower()[:6], 41212)))
    tokens = set(tokens)
    hashes = []
    for i in range(10):
        l = [abs(mmh3.hash(str(t), 232323+i)) for t in tokens]
        hashes.append(np.min(l))
    return hashes
def get_random_tokens(id):
    hashes = []
    for i in range(10):
        hashes.append(abs(mmh3.hash(str(id), 2323211+i)))
def get_random_tokens(id):
    hashes = []
    for i in range(10):
        hashes.append(abs(mmh3.hash(str(id), 2323211+i)))
    return np.array(hashes)
import re
def get_tokens(s):
    tokens = []
    for token in re.split('[, :)(]', s):
        tokens.append(abs(mmh3.hash(token.lower()[:6], 41212)))
    tokens = set(tokens)
    hashes = []
    for i in range(10):
        l = [abs(mmh3.hash(str(t), 232323+i)) for t in tokens]
        hashes.append(np.min(l))
    return np.array(hashes)
import re
def get_tokens(s):
    tokens = []
    for token in re.split('[, :)(]', s):
        tokens.append(abs(mmh3.hash(token.lower()[:6], 41212)))
    tokens = set(tokens)
    hashes = []
    for i in range(10):
        l = [abs(mmh3.hash(str(t), 232323+i)) for t in tokens]
        hashes.append(np.min(l))
    return np.array(hashes)
for i in range(hashes.shape[0])
tags['movieId']
0 in tags['movieId']
1 in tags['movieId']
1 in tags['movieId'].values
0 in tags['movieId'].values
minhashes = np.zeros(hashes.shape[0], 10)
for i in range(hashes.shape[0]):
    if i in tags['movieId'].values:
        minhashes[i] = get_tokens(tags[tags['movieId']==i]['tag'])
    else:
        minhashes[i] = get_random_tokens(i)
minhashes = np.zeros(hashes.shape[0], 10)
for i in range(hashes.shape[0]):
    if i in tags['movieId'].values:
        minhashes[i] = get_tokens(tags[tags['movieId']==i]['tag'].values[0])
    else:
        minhashes[i] = get_random_tokens(i)
minhashes = np.zeros((hashes.shape[0], 10), dtype=np.int32)
for i in range(hashes.shape[0]):
    if i in tags['movieId'].values:
        minhashes[i] = get_tokens(tags[tags['movieId']==i]['tag'].values[0])
    else:
        minhashes[i] = get_random_tokens(i)
minhahes
minhashes
minhashes
minhashes.shape
tags
minhashes[260]
minhashes[1127]
minhashes[2420]
minhashes[2421]
minhashes[2422]
np.savez_compressed?
np.savez_compressed("minhashes_int32_10.npz", hashes=minhashes)
history

