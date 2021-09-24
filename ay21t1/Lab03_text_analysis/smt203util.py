from nltk.corpus import stopwords as nltk_stopwords
import numpy as np 
import pandas as pd
import csv
import os
import re
from os import path
import urllib.request
import io

__version__ = 0.1


def get_stopwords():
    '''selecting words which appear only in the web text corpus and not in stop words'''
    stopwords_list = list(nltk_stopwords.words('english'))

    return stopwords_list

def read_word_count_file_online(target_url):
    
    data = urllib.request.urlopen(target_url)
    reader = csv.reader(io.TextIOWrapper(data))
    counts = {rows[0]: int(rows[1]) for rows in reader}
    
    return counts

def read_word_count_file(file_path):
    with open(file_path, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        counts = {rows[0]: float(rows[1]) for rows in reader}
    return counts


def calculate_log_odds_idp(global_counts, counts_i_name, counts_i_dict, counts_j_name, counts_j_dict):
    '''calculate log odds'''
    global_df = pd.DataFrame(list(global_counts.items()), columns=['word', 'global_count'])

    global_df[counts_i_name] = global_df.word.apply(lambda word: counts_i_dict[word]
    if word in counts_i_dict.keys() else 0)
    global_df[counts_j_name] = global_df.word.apply(lambda word: counts_j_dict[word]
    if word in counts_j_dict.keys() else 0)
    
    # Log odds ratio infromatice Dirichlet prior calculation
    # hate -> i, ref -> j
    ni = global_df[counts_i_name].sum()
    nj = global_df[counts_j_name].sum()
    a0 = global_df.global_count.sum()

    yiw = global_df[counts_i_name]
    yjw = global_df[counts_j_name]
    aw = global_df.global_count

    sd = (1 / (yiw + aw) + 1 / (yjw + aw)).apply(np.sqrt)

    delta = (((yiw + aw) / (ni + a0 - (yiw + aw))) / ((yjw + aw) / (nj + a0 - (yjw + aw)))).apply(np.log)

    global_df['log_odds_z_score'] = delta / sd

    global_df = global_df.sort_values(by=['log_odds_z_score'], ascending=False)

    return global_df

def find_discriminative_words(top_words_df, threshold_i=5, threshold_j=5, num_i=100, num_j=100, mypath='.'):
    '''write discriminative words to each file separately'''
    counts_i_name = top_words_df.columns[2]
    counts_j_name = top_words_df.columns[3]
    
    # tmp = top_words_df[top_words_df[counts_j_name] >= threshold_j].query('log_odds_z_score > 0').head(num_j)
    tmp = top_words_df[top_words_df[counts_j_name] >= threshold_j].head(num_j)
    
    with open(f"{mypath}/word_counts/wordcloud_word_{counts_i_name}_zscore.csv", "w") as output:
        for index, row in tmp.iterrows():
            output.write(",".join(["_".join(row['word'].split()), str(-1.0*row['log_odds_z_score'])])+"\n")

    # tmp = top_words_df[top_words_df[counts_i_name] >= threshold_i].query('log_odds_z_score < 0').iloc[::-1].head(num_i)
    tmp = top_words_df[top_words_df[counts_i_name] >= threshold_i].iloc[::-1].head(num_i)
    with open(f"{mypath}/word_counts/wordcloud_word_{counts_j_name}_zscore.csv", "w") as output:
        for index, row in tmp.iterrows():
            output.write(",".join(["_".join(row['word'].split()), str(row['log_odds_z_score'])])+"\n")
