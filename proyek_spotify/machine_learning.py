# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 13:19:15 2023

@author: Ditto Ryanda
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import preprocessing
from statsmodels.stats.outliers_influence import variance_inflation_factor
from yellowbrick.cluster import SilhouetteVisualizer
import pickle
from scipy.spatial import distance
from scipy.spatial.distance import cdist
import seaborn as sns
import math
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('C:\Akademik\Proyek DS\dataset_komplit.csv') 
df.drop(['Unnamed: 0'], axis = 1, inplace=True)
df.drop_duplicates(subset='uri', inplace=True)

# df.info()

# Check null and duplicate values
# print('Null values : \n', df.isnull().sum())
# print('Duplicate values : ', df.duplicated().sum())

# Selected predictors (based on domain knowledge)
df_audio_features = df[['name',
                        'energy',
                        'loudness',
                        'speechiness',
                        'acousticness',
                        'instrumentalness',
                        'valence',
                        'tempo',
                        'danceability',
                        'liveness',
                        'year',
                        'popularity']].copy()

df_audio_features.drop_duplicates(inplace=True)

# y = df_audio_features['popularity'] # Dependent var
# X_linear_reg = df_audio_features[['danceability', 
#                         'energy',
#                         'loudness',
#                         'speechiness',
#                         'acousticness',
#                         'instrumentalness',
#                         'liveness',
#                         'valence',
#                         'tempo']] # Predictor
# X_linear_reg = sm.add_constant(X_linear_reg)

# Check the coef corr of popularity with audio features
# All audio features have low coef corr with popularity
# corr = df_audio_features.corr()
# plt.figure(figsize=(20,20))
# sns.heatmap(corr, annot=True)

# Linear regression
# model = sm.OLS(y, X_linear_reg)

# Result
# R-Squared value 0.032 which means the model can only affect the dependent var (popularity) for 3.2%
# result = model.fit()
# print(result.summary())

# Check multicollinearity
# Very high VIF in danceability, energy, loudness, valence, tempo indicates multicollinearity
# def calc_vif(X) :
#     vif = pd.DataFrame()
#     vif['variables'] = X.columns
#     vif['vif'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
#     return vif
# print(calc_vif(X.drop(columns='const')))

# Check the distribution of residuals
# sns.displot(result.resid)

#---------------------------------------------------------------
# KMeans
X_kmeans = df_audio_features[['name',
                        'energy',
                        'loudness',
                        'speechiness',
                        'acousticness',
                        'instrumentalness',
                        'valence',
                        'tempo',
                        'danceability',
                        'year',
                        'popularity']].copy()

# wcss = []
# for i in range(1,31):
#     k_means = KMeans(n_clusters=i, init='k-means++', random_state=69, n_init='auto')
#     k_means.fit(X_kmeans.drop(['name','year','popularity'], axis=1))
#     print('K = {}, Inertia = {}'.format(i, k_means.inertia_))
#     wcss.append(k_means.inertia_)
      
# plt.plot(range(1,31), wcss, marker ='o')
# plt.xlabel('Clusters Count', fontsize='20')
# plt.ylabel('Inertia', fontsize='20')
# plt.show()

k_means_optimum = KMeans(n_clusters=5, 
                          init='k-means++',
                          random_state=42, 
                          n_init='auto')

# visualizer = SilhouetteVisualizer(k_means_optimum, colors='yellowbrick')
# visualizer.fit(X.drop(['popularity'], axis=1))
# visualizer.show() 

y = k_means_optimum.fit_predict(X_kmeans.drop(['name','year','popularity'], axis=1))
X_kmeans['cluster'] = y

iqr_kmeans = []
for i in sorted(list(pd.unique(X_kmeans['cluster']))) :
    popularity_cluster = X_kmeans[X_kmeans['cluster']==i]['popularity']
    iqr_cluster = popularity_cluster.quantile(0.75)-popularity_cluster.quantile(0.25)
    iqr_kmeans.append(int(iqr_cluster))

plt.figure(figsize=(12,8))
plot = sns.histplot(iqr_kmeans)
plot.set_xlim(1, max(iqr_kmeans)+1)
plot.set_xticks(range(0, max(iqr_kmeans)+1))
plt.xlabel('Interquartile Range', fontsize='20')
plt.ylabel('Frequency', fontsize='20')
plt.title('Persebaran Nilai Popularitas dalam Cluster', fontsize='25')

#---------------------------------------------------------------
# Agglomerative Clustering

X = df_audio_features[['name',
                        'energy',
                        'loudness',
                        'speechiness',
                        'acousticness',
                        'instrumentalness',
                        'valence',
                        'tempo',
                        'danceability',
                        'year',
                        'popularity']].copy()

# normalized_tempo = np.array(X['tempo'])
# normalized_tempo = preprocessing.normalize([normalized_tempo], norm='max')
# X['tempo'] = normalized_tempo[0]

# normalized_loudness = np.array(X['loudness'])
# normalized_loudness = preprocessing.normalize([normalized_loudness], norm='max')
# X['loudness'] = normalized_loudness[0]

# min_distance = []
# def min_distance_sample(n) :
#     sample = X.drop(['name','year','popularity'], axis=1).sample(n)
#     distances = cdist(sample, sample, 'euclidean')
#     output = []
#     i, j = [0,0]
#     while i < len(distances):
#         j = 0
#         i_distances = []
#         while j < i:
#             i_distances.append(distances[i][j])
#             j += 1
#         if i != 0 :
#             output.append(np.quantile(np.array(i_distances), 0.25))
#             # output.append(min(np.array(i_distances)))
#         i += 1
#     output = np.array(output)
    
#     #returns a list of minimum distance of every data point.
#     # return np.quantile(output, 0.5) 
#     # return min(output)
#     return output.mean()

# for i in range(0, 10) :
#     min_distance.append(min_distance_sample(int(len(X)/10)))

# threshold = np.array(min_distance).mean()
# threshold = np.quantile(np.array(min_distance), 0.5)

# agglo_model = AgglomerativeClustering(affinity='euclidean',
#                                       linkage='complete',
#                                       n_clusters = None,
#                                       distance_threshold=threshold)

# agglo_result = agglo_model.fit_predict(X.drop(['name','year','popularity'], axis=1))

# X_agglo = X.copy()
# X_agglo['cluster'] = agglo_result

X_agglo = pickle.load(open('X_agglo_result', 'rb'))

# See the distribution of popularity every cluster
iqr = []
q1 = []
q3 = []
for i in sorted(list(pd.unique(X_agglo['cluster']))) :
    popularity_cluster = X_agglo[X_agglo['cluster']==i]['popularity']
    iqr_cluster = popularity_cluster.quantile(0.75)-popularity_cluster.quantile(0.25)
    iqr.append(int(iqr_cluster))
    
    q1.append(popularity_cluster.quantile(0.25))
    q3.append(popularity_cluster.quantile(0.75))

plt.figure(figsize=(12,8))
plot = sns.histplot(iqr)
plot.set_xlim(1, max(iqr)+1)
plot.set_xticks(range(0, max(iqr)+1, 2))
plt.xlabel('Interquartile Range', fontsize='20')
plt.ylabel('Frequency', fontsize='20')
plt.title('Persebaran Popularitas Per Cluster')

# Predict 
X_test_agglo = pd.read_excel(r'C:\Users\regis\Downloads\2023.xlsx')
X_test_agglo.drop_duplicates(subset='uri', inplace=True)
X_test_agglo['year'] = 2023
X_test_agglo = X_test_agglo[['name',
                            'energy',
                            'loudness',
                            'speechiness',
                            'acousticness',
                            'instrumentalness',
                            'valence',
                            'tempo',
                            'danceability',
                            'year',
                            'popularity']].copy()
normalized_tempo_test = np.array(X_test_agglo['tempo'])
normalized_tempo_test = preprocessing.normalize([normalized_tempo_test], norm='max')
X_test_agglo['tempo'] = normalized_tempo_test[0]

normalized_loudness_test = np.array(X_test_agglo['loudness'])
normalized_loudness_test = preprocessing.normalize([normalized_loudness_test], norm='max')
X_test_agglo['loudness'] = normalized_loudness_test[0]
# pickle.dump(X_test_agglo, open('test_agglo', 'wb'))

# KNN for classifying cluster of test data
knn = KNeighborsClassifier(n_neighbors=1, 
                           metric='euclidean',
                           weights='distance',
                           algorithm='brute')
knn.fit(X_agglo.drop(['name','year','popularity','cluster'], axis=1), X_agglo['cluster'])
y_pred = knn.predict(X_test_agglo.drop(['name','year','popularity'], axis=1))

X_test_agglo['cluster'] = y_pred

# Evaluate Prediction based on popularity
cluster_stats = pd.DataFrame(columns=['cluster', 'q1', 'q3', 'iqr'])
cluster_stats['cluster'] = sorted(list(pd.unique(X_agglo['cluster'])))
cluster_stats['q1'] = q1
cluster_stats['q3'] = q3
cluster_stats['iqr'] = iqr

# Accuracy
true_pred = 0
total_pred = len(X_test_agglo)
for idx, pred in X_test_agglo[['cluster','popularity']].iterrows() :
    if pred[1] >= list(cluster_stats[cluster_stats['cluster']==pred[0]]['q1'])[0] :
        if pred[1] <= list(cluster_stats[cluster_stats['cluster']==pred[0]]['q3'])[0] :
            true_pred += 1
accuracy = true_pred/total_pred



year = list(range(1960,2022))
year.remove(2005)

zero_dominance = []
for i in year :
    zero_dominance.append(
        len(X[(X['year']==i) & (X['popularity']==0)])
                          /len(X[X['year']==i]))

# count = 0
# for element in zero_dominance :
#     if element > 0.75 :
#         count += 1

# plt.bar(year, iqr)
# plt.title('iqr')
# plt.show()

iqr_year = []
for i in year :
    popularity_year = X[X['year']==i]['popularity']
    iqr_year.append(popularity_year.quantile(0.75)-popularity_year.quantile(0.25))

plt.figure(figsize=(12,8))
plt.plot(year, zero_dominance, linewidth=5)
plt.title('Persentase Nilai 0 Per Tahun')
plt.xlim(range(1960,2022,10))
plt.show()

plt.figure(figsize=(12,8))
plt.plot(year, iqr_year, 'r-', linewidth=5)
plt.title('Persebaran Popularitas Per Tahun')
plt.xlim(range(1960,2022,10))
plt.show()
