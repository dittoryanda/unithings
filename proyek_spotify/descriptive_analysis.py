# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

df = pd.read_csv('C:\Akademik\Proyek DS\dataset_komplit.csv') 
df.drop(['Unnamed: 0'], axis = 1, inplace=True)

artist_name = []

for artist in df['artists'].values:
    artist_to_json = json.loads(artist)
    dict_artist = artist_to_json[0]
    artist_name.append(dict_artist['name'])
    
df['artist_name'] = artist_name

#################################################################################################

# DataFrame untuk menyimpan seluruh nilai median dari setiap fitur audio per tahun (1 row = 1 tahun)
df_popular = pd.DataFrame(columns=['energy', 
                                   'valence', 
                                   'tempo', 
                                   'loudness', 
                                   'instrumentalness', 
                                   'speechiness', 
                                   'acousticness', 
                                   'danceability', 
                                   'year'])

for i in range(1960, 2023):
    if i == 2005 :
        continue
    else :
        df_year = df[df['year']==i] # Mengambil data dari tahun i
        df_popular_per_year = df_year[df_year['popularity']>= 20] # Mengambil lagu-lagu yang populer dari tahun i
        # print(df_popular_per_year['popularity'].min(axis=0), '\t', i, '\t', len(df_popular_per_year)) <debugging>
        df_new_row = pd.DataFrame({'energy':[df_popular_per_year['energy'].median()], 
                                   'valence':[df_popular_per_year['valence'].median()], 
                                   'tempo':[df_popular_per_year['tempo'].median()],
                                   'loudness':[df_popular_per_year['loudness'].median()], 
                                   'acousticness':[df_popular_per_year['acousticness'].median()],
                                   'danceability':[df_popular_per_year['danceability'].median()],
                                   'instrumentalness':[df_popular_per_year['instrumentalness'].median()],
                                   'speechiness':[df_popular_per_year['speechiness'].median()],
                                   'year':[i]}) # Menyimpan nilai median dari setiap fitur audio pada tahun i
        df_popular = pd.concat([df_popular, df_new_row], ignore_index=True) # Menyimpan nilai median setiap fitur audio pada tahun i ke DataFrame utama nilai median seluruh tahun
        
#################################################################################################
# Line Plot Energy

plt.figure(figsize=(10,5))
z = np.polyfit(list(df_popular['year']), list(df_popular['energy']), 1)
p = np.poly1d(z)
plt.plot(df_popular['year'], p(df_popular['year']))
plt.plot(df_popular['year'], df_popular['energy'], marker="o", color='red')
# plt.ylim([0, 1])
plt.title("Grafik Energy 1960-2022", fontsize = 20)
plt.xlabel("Tahun")
plt.ylabel("Energy")
plt.xticks(range(1960,2022,4))
plt.grid()
plt.show()

#################################################################################################
# Line Plot Valence

plt.figure(figsize=(10,5))
z = np.polyfit(list(df_popular['year']), list(df_popular['valence']), 1)
p = np.poly1d(z)
plt.plot(df_popular['year'], p(df_popular['year']))
plt.plot(df_popular['year'], df_popular['valence'], marker="o")
# plt.ylim([0, 1])
plt.title("Grafik Valence 1960-2022", fontsize = 20)
plt.xlabel("Tahun")
plt.ylabel("Valence")
plt.xticks(range(1960,2022,4))
plt.grid()
plt.show()

#################################################################################################
# Line Plot Tempo

plt.figure(figsize=(10,5))
z = np.polyfit(list(df_popular['year']), list(df_popular['tempo']), 1)
p = np.poly1d(z)
plt.plot(df_popular['year'], p(df_popular['year']))
plt.plot(df_popular['year'], df_popular['tempo'], marker="o", color='green')
# plt.ylim([0, 1])
plt.title("Grafik Tempo 1960-2022", fontsize = 20)
plt.xlabel("Tahun")
plt.ylabel("Tempo (BPM)")
plt.xticks(range(1960,2022,4))
plt.grid()
plt.show()

#################################################################################################
# Line Plot Loudness

plt.figure(figsize=(10,5))
z = np.polyfit(list(df_popular['year']), list(df_popular['loudness']), 1)
p = np.poly1d(z)
plt.plot(df_popular['year'], p(df_popular['year']))
plt.plot(df_popular['year'], df_popular['loudness'], marker="o", color='black')
# plt.ylim([0, 1])
plt.title("Grafik Loudness 1960-2022", fontsize = 20)
plt.xlabel("Tahun")
plt.ylabel("Loudness (dB)")
plt.xticks(range(1960,2022,4))
plt.grid()
plt.show()

#################################################################################################
# Line Plot Instrumentalness

# plt.figure(figsize=(8,5))
# plt.plot(df_popular['year'], df_popular['instrumentalness'], marker="o", color='black')
# # plt.ylim([0, 1])
# plt.title("Grafik Tren Instrumentalness dari Lagu-lagu Populer Tahun 1960-2022")
# plt.xlabel("Tahun")
# plt.ylabel("Instrumentalness")
# plt.xticks(range(1960,2022,4))
# plt.grid()
# plt.show()

#################################################################################################
# # Line Plot Speechiness

# plt.figure(figsize=(8,5))
# plt.plot(df_popular['year'], df_popular['speechiness'], marker="o", color='black')
# # plt.ylim([0, 1])
# plt.title("Grafik Tren Speechiness dari Lagu-lagu Populer Tahun 1960-2022")
# plt.xlabel("Tahun")
# plt.ylabel("Speechiness")
# plt.xticks(range(1960,2022,4))
# plt.grid()
# plt.show()

#################################################################################################
# Line Plot Acousticness

plt.figure(figsize=(10,5))
z = np.polyfit(list(df_popular['year']), list(df_popular['acousticness']), 1)
p = np.poly1d(z)
plt.plot(df_popular['year'], p(df_popular['year']))
plt.plot(df_popular['year'], df_popular['acousticness'], marker="o", color='orange')
# plt.ylim([0, 1])
plt.title("Grafik Acousticness 1960-2022", fontsize = 20)
plt.xlabel("Tahun")
plt.ylabel("Acousticness")
plt.xticks(range(1960,2022,4))
plt.grid()
plt.show()

#################################################################################################
# # Line Plot Danceability

# plt.figure(figsize=(8,5))
# plt.plot(df_popular['year'], df_popular['danceability'], marker="o", color='black')
# # plt.ylim([0, 1])
# plt.title("Grafik Tren Danceability dari Lagu-lagu Populer Tahun 1960-2022")
# plt.xlabel("Tahun")
# plt.ylabel("Danceability")
# plt.xticks(range(1960,2022,4))
# plt.grid()
# plt.show()



#################################################################################################

corr = df_popular[['energy', 'valence', 'tempo', 'loudness', 'acousticness']].corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, annot=True)
plt.show()

#################################################################################################

# Debugging
# df_popular.to_csv('C:\\Akademik\\Proyek DS\\viz_prods_question_2.csv', index=False)
