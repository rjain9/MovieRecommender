import pandas as pd
import numpy as np
import math, copy
import re
from surprise import Reader, Dataset, SVD, evaluate
from scipy.sparse.linalg import svds
import operator

# path to where the input file is present
path = './'

# Reference: https://www.kaggle.com/laowingkin/netflix-movie-recommendation
df = pd.read_csv(path + 'NewMergedFile.txt', header = None, names = ['Cust_Id', 'Rating', 'Movie_Id'], usecols = [0,1,3])
df['Rating'] = df['Rating'].astype(float)
df['Movie_Id'] = df['Movie_Id'].astype(int)
df['Cust_Id'] = df['Cust_Id'].astype(int)

# print('Dataset 1 shape: {}'.format(df.shape))
# print('-Dataset examples-')
# print(df.iloc[::5000000, :])


# DATA CLEANING
df.index = np.arange(0,len(df))
df_nan = pd.DataFrame(pd.isnull(df.Rating))
df_nan = df_nan[df_nan['Rating'] == True]
df_nan = df_nan.reset_index()

# Data slicing
f = ['count','mean']

df_movie_summary = df.groupby('Movie_Id')['Rating'].agg(f)
df_movie_summary.index = df_movie_summary.index.map(int)
movie_benchmark = round(df_movie_summary['count'].quantile(0.2),0)
drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

# print('Movie minimum times of review: {}'.format(movie_benchmark))

df_cust_summary = df.groupby('Cust_Id')['Rating'].agg(f)
df_cust_summary.index = df_cust_summary.index.map(int)
cust_benchmark = round(df_cust_summary['count'].quantile(0.2),0)
drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index

# print('Customer minimum times of review: {}'.format(cust_benchmark))

# Removing Data
# print('Original Shape: {}'.format(df.shape))
df = df[~df['Movie_Id'].isin(drop_movie_list)]
df = df[~df['Cust_Id'].isin(drop_cust_list)]
# print('After Trim Shape: {}'.format(df.shape))
# print('-Data Examples-')
# print(df.iloc[::5000000, :])

df_p = pd.pivot_table(df,values='Rating',index='Cust_Id',columns='Movie_Id')

# Reading movies data from the movie_titles.txt file
df_title = pd.read_csv(path + 'movie_titles.txt', encoding = "ISO-8859-1", header = None, names = ['Movie_Id', 'Year', 'Name'])
df_title.set_index('Movie_Id', inplace = True)
# print (df_title.head(10))

# Reference: https://github.com/vdyashin/SVD/blob/master/function.py
def SVD(mat, initial_mat1, initial_mat2, learn_rate, iterations):
    A = np.array(mat)
    # two matrices from which we will start: B is m by k
    B = np.array(initial_mat1)
    # C is n by k
    C = np.array(initial_mat2)
    # learning rate, or step of learning
    alpha = learn_rate
    # number of iterations
    N = iterations
    # A ~ B * C^t : the first approximation based on given initial matrices
    A_app = np.dot(B, C.T)
    # gradient descent
    for i in range(N):
        # partial derivatives for matrices

        dLdB = np.dot((A_app - A), C)
        # dLdB = np.nansum((A_app - A)* C)
        dLdC = np.dot((A_app - A).T, B)
        # dLdC = np.nansum((A_app - A).T* B)
        # updating matrices
        C = C - alpha * dLdC
        B = B - alpha * dLdB
        # calculating approximated matrix
        A_app = np.dot(B, C.T)
    return B, C, A_app


def euclideanDist(x, xi):
    d = 0.0
    for i in range(len(x)-1):
        d += pow((float(x[i])-float(xi[i])),2)
    # d = math.sqrt(d)
    return d

def knn_predict(input_movie_index, train_data, k_value=10):
    dist = {}

    input_movie_row = train_data[input_movie_index]
    # print input_movie_index, input_movie_row
    for index, row in enumerate(train_data):
        dist[index] = euclideanDist(row, input_movie_row)

    # print dist
    sorted_dist = sorted(dist.items(), key=operator.itemgetter(1))
    rv = []
    # print sorted_dist[:10]
    for item in xrange(min(k_value, len(sorted_dist))):
        rv.append(sorted_dist[item][0])

    return rv

def test(movieUserMatrix, movieCount, custIdCount):
    # print movieUserMatrix
    k = 4
    B = []
    for row in xrange(movieCount):
        temp = [0.1]*k
        B.append(temp)

    C = []
    for row in xrange(custIdCount):
        temp = [0.1]*k
        C.append(temp)
    
    # learning rate
    alpha = 0.001
    # number of iterations
    N = 1000

    # print 'row col'
    # print len(movieUserMatrix), len(movieUserMatrix[0])
    originalMatrix = copy.deepcopy(movieUserMatrix)
    i = int(len(movieUserMatrix)*0.4)
    j = int(len(movieUserMatrix[0])*0.4)
    for x in xrange(i):
        for y in xrange(j):
            movieUserMatrix[x][y] = 0
    B, C, newMovieUserMatrix = SVD(mat = movieUserMatrix, initial_mat1 = B, initial_mat2 = C, learn_rate = alpha, iterations = N)

    rmse = 0
    count = 0
    for x in xrange(i):
        for y in xrange(j):
            if int(originalMatrix[x][y]) != 0:
                if newMovieUserMatrix[x][y] > 2.5:
                    # print originalMatrix[x][y], newMovieUserMatrix[x][y]
                    rmse += ((originalMatrix[x][y] - newMovieUserMatrix[x][y]) * (originalMatrix[x][y] - newMovieUserMatrix[x][y]))
                    count +=1
    # print "rmse: ", rmse, "count: ", count
    if count != 0:
        print "rmse: ", math.sqrt(rmse/count)

def knn_recommend(df, movie_title, printFlag = True):
    movieIds = {}
    custIds = {}
    movieCount = 0
    custIdCount = 0
    for index, row in df.iterrows():
        temp_Movie_Id = int(row['Movie_Id'])
        temp_Cust_Id = int(row['Cust_Id'])
        # print 'Movie_Id', temp_Movie_Id, 'Cust_Id', temp_Cust_Id
        if temp_Movie_Id not in movieIds:
            movieIds[temp_Movie_Id] = movieCount
            movieCount +=1
        if temp_Cust_Id not in custIds:
            custIds[temp_Cust_Id] = custIdCount
            custIdCount +=1

    # print 'No. of movies: ', movieCount
    # print 'No. of Users: ', custIdCount
    movieUserMatrix = []
    for row in xrange(movieCount):
        temp = [0.0]*custIdCount
        movieUserMatrix.append(temp)

    for index, row in df.iterrows():
        movieUserMatrix[movieIds[row['Movie_Id']]][custIds[row['Cust_Id']]] = row['Rating']

    # print movieUserMatrix
    k = 4
    B = []
    for row in xrange(movieCount):
        temp = [0.1]*k
        B.append(temp)

    C = []
    for row in xrange(custIdCount):
        temp = [0.1]*k
        C.append(temp)
    
    # learning rate
    alpha = 0.001
    # number of iterations
    N = 1000

    test(copy.deepcopy(movieUserMatrix), movieCount, custIdCount)
    B, C, newMovieUserMatrix = SVD(mat = movieUserMatrix, initial_mat1 = B, initial_mat2 = C, learn_rate = alpha, iterations = N)
    # print newMovieUserMatrix, B.size, C.size
    input_movie_index = int(df_title.index[df_title['Name'] == movie_title][0])
    # print input_movie_index
    # print 'movieIds', movieIds
    recommendedMovieIds = knn_predict(movieIds[input_movie_index], B)
    # print recommendedMovieIds
    mIDs = []
    for movieId in recommendedMovieIds:
        mIDs.append(movieIds.keys()[movieIds.values().index(movieId)] - 1)
    
    if printFlag:
        print("For movie ({})".format(movie_title)) 
        print("- Top 10 movies recommended based on KNN - ")
        print df_title.iloc[mIDs]
    return df_title.iloc[mIDs]


# Recommendation according to the Pearsons'R correlation
def recommend(movie_title, min_count, printFlag = True):
    if printFlag:
        print("For movie ({})".format(movie_title))
        print("- Top 10 movies recommended based on Pearsons'R correlation - ")
    i = int(df_title.index[df_title['Name'] == movie_title][0])
    target = df_p[i]
    similar_to_target = df_p.corrwith(target)
    corr_target = pd.DataFrame(similar_to_target, columns = ['PearsonR'])
    corr_target.dropna(inplace = True)
    corr_target = corr_target.sort_values('PearsonR', ascending = False)
    corr_target.index = corr_target.index.map(int)
    corr_target = corr_target.join(df_title).join(df_movie_summary)[['PearsonR', 'Name', 'count', 'mean']]
    if printFlag:
        print(corr_target[corr_target['count']>min_count][:10].to_string(index=False))
    return corr_target[corr_target['count']>min_count][:10]

def checkRecommendationAccuracy(movieRecco, df, movie, algo = 'KNN'):
    count = 0
    for index, row in movieRecco.iterrows():
        # print row['Name']
        if algo == 'KNN':
            recco = knn_recommend(df, row['Name'], False)
        else:
            recco = recommend(row['Name'], 0, False)
        for i, inner_row in recco.iterrows():
            if inner_row['Name'] == movie:
                # print inner_row['Name']
                count +=1
                break

    print "The input movie appeared in the recommendation of recommended movies ", count, "times"

import sys
movie = "Justice League"
if len(sys.argv) > 1:
    movie = sys.argv[1]
recco = knn_recommend(df, movie)
checkRecommendationAccuracy(recco, df, movie, 'KNN')
recco = recommend(movie, 0)
checkRecommendationAccuracy(recco, df, movie, 'PC')