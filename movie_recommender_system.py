import numpy as np
import nltk
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("C:\\Users\\ARADHYA GARG\\Desktop\\Movie_Recommender_System\\tmdb_5000_credits.csv (2) (1).zip")
cred = pd.read_csv("C:\\Users\\ARADHYA GARG\\Desktop\\Movie_Recommender_System\\tmdb_5000_movies.csv (1).zip")
# print(movies.head(1))
# print(credits.head(1)['cast'].values)
# print(movies.merge(credits,on='title').shape)
movies = movies.merge(cred, on='title')
# just fetcth the important columns from the dataframe
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
# print(movies.isnull().sum())    #we got three movies whose data was not there
movies.dropna(inplace=True)  # here we drop them


# print(movies.isnull().sum())
# now check fro duplicasy
# print(movies.duplicated().sum())  # now we did not get any duplicasy as the result is 0


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


movies['genres'] = movies['genres'].apply(convert)
# print(movies.iloc[0].genres)
movies['keywords'] = movies['keywords'].apply(convert)


# print(movies.head())

# remove the unwanted crew members from the dataframe
def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter = counter + 1
        else:
            break
    return L


movies['cast'] = movies['cast'].apply(convert3)


# print(movies.info())
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())
# remove the spaces fromt hte names of the charaters and the directors
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['movie_id', 'title', 'tags']]
# print(new_df.head())
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
# print(new_df['tags'][0])
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
# print(new_df.head())
# print(cv.get_feature_names())

ps=PorterStemmer()
def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))

    return " ".join

new_df['tags']=new_df['tags'].apply(stem)
cv=CountVectorizer(max_features=5000,stop_words='english')
vectors=cv.fit_transform(new_df['tags']).toarray()

# print(cv.get_feature_names())
similarity=cosine_similarity(vectors)
# print(similarity)

def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)

recommend('Avatar')

# new_df.iloc[1216].title