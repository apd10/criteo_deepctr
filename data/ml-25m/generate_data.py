import pandas as pd
import numpy as np
from sklearn.utils import shuffle

SMALL=False
suffix="-short"
ratings = pd.read_csv("ratings.csv")
if SMALL:
    ratings = ratings.head(1000)

movies = pd.read_csv("movies.csv")
genres_list= ["Action",
  "Adventure",
  "Animation",
  "Children",
  "Comedy",
  "Crime",
  "Documentary",
  "Drama",
  "Fantasy",
  "Horror",
  "IMAX",
  "Musical",
  "Mystery",
  "Romance",
  "Sci-Fi",
  "Thriller",
  "War",
  "listed"]


for i in range(len(genres_list)):
    gen = genres_list[i]
    print(gen)
    movies[gen] = movies['genres'].str.contains(gen).astype(int)

del movies['genres']
# not using title
del movies['title']

#genome_scores = pd.read_csv("genome-scores.csv")
#genome_scores  = genome_scores.pivot_table(values="relevance", index="movieId", columns="tagId").reset_index()
#genome_scores = genome_scores.rename(columns={i+1:'G'+str(i+1) for i in range(1128)})


data = pd.merge(left=ratings, right=movies, how="left", on="movieId")
#data = pd.merge(left=data, right=genome_scores, how="left", on="movieId")

column_info = np.array(data.columns)
np_data = data.to_numpy()
num_samples = np_data.shape[0]
idx = np.arange(num_samples)
np.random.shuffle(idx)
np_data = np_data[idx]

train_num_samples = int(0.7 * num_samples)
test_num_samples = num_samples - train_num_samples


train_data = np_data[:train_num_samples]
test_data = np_data[train_num_samples:]

if SMALL:
    np.savez("data-small"+suffix+".npz", train=train_data, test=test_data)
else:
    np.savez("data"+suffix+".npz", train=train_data, test=test_data)
np.savetxt("columns"+suffix+".txt", column_info, fmt="%s") 
