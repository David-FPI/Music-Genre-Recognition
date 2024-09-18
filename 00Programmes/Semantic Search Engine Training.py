
#######################################################################################################
############################## CODE AUTHOR: DAVID ISSÁ                  ###############################
############################## DATE: September 2024                     ###############################
############################## SUBJECT: SEMANTIC SEARCH ENGINE TRAINING ###############################
#######################################################################################################


#######################################################################################################
###################################             READ ME             ###################################
#######################################################################################################



#######################################################################################################
###################################       PROGRAMME AUTOMATION      ###################################
#######################################################################################################

import pandas as pd
import numpy as np
import csv
from datetime import datetime
import os
import re
import pyodbc
from sentence_transformers import SentenceTransformer
import torch
from joblib import Parallel, delayed
import re
import warnings
import random
random.seed(99)
warnings.filterwarnings("ignore")

source_path = r'C:\Users\david\OneDrive\Ambiente de Trabalho\ITC\03SemanticSearchEngine\01Source'

exit_path = r'C:\Users\david\OneDrive\Ambiente de Trabalho\ITC\03SemanticSearchEngine\02Exit'

end_path = r'C:\Users\david\OneDrive\Ambiente de Trabalho\ITC\03SemanticSearchEngine\03End'

wd_path = r'C:\Users\david\OneDrive\Ambiente de Trabalho\ITC\03SemanticSearchEngine\00Programmes'

os.chdir(wd_path)


#######################################################################################################
##############################      0. IMPORTING SOURCE DATASETS      ################################
#######################################################################################################

# Connect to SQL server
connection_string = ("driver={ODBC Driver 17 for SQL Server};"
                     "Server=DAVID;"
                     "Database=HS_Search;"
                     "UID=sa;"
                     "PWD=19071999")

connection = pyodbc.connect(connection_string)
cursor = connection.cursor()

# Import necessary databases
arc_df = pd.read_sql("SELECT * FROM dbo.Arc", connection)
simplex_df = pd.read_sql("SELECT * FROM dbo.Simplex", connection)
ntlc_df = pd.read_sql("SELECT * FROM dbo.zzz_Ntlc", connection)
hs_df = pd.read_sql("SELECT * FROM dbo.zzz_HS", connection)



#######################################################################################################
###########################         1. PREPROCESSING SOURCE DATASET         ###########################
#######################################################################################################

# Add ntlc to arc table based on Ntlc_ID
arc_df = arc_df[['PdtName','PdtSpecification','Ntlc_ID']]
arc_df['descp_type'] = 'arc_desc'
arc_df = arc_df.merge(ntlc_df, how='left', left_on = 'Ntlc_ID', right_on='Ntlc_id').drop(columns=['Ntlc_ID','Ntlc_id'])


# Add ntlc to simplex table based on HS_ID
simplex_df = simplex_df[simplex_df['Type_ID'].isin([1, 2, 4])].reset_index(drop=True)
simplex_df = simplex_df.rename(columns={"SimplexDesc": "PdtName"})
type_mapping = {1: 'se_desc', 2: 'simplex_desc', 4: 'chem_desc'}
simplex_df['descp_type'] = simplex_df['Type_ID'].map(type_mapping)
simplex_df['PdtSpecification'] = simplex_df['PdtName']
simplex_df = simplex_df[['PdtName','PdtSpecification','HS_ID','descp_type']]

hs_df = hs_df[['HS_ID','HS_Code']].drop_duplicates().reset_index(drop=True)
simplex_df = simplex_df.merge(hs_df, how='left', on = 'HS_ID').drop(columns='HS_ID').rename(columns={'HS_Code': 'Ntlc'})


# Final dataset, combining both arc and simplex datasets
final_df = pd.concat([arc_df,simplex_df]).reset_index(drop=True)


# Create hs6 fetaure based on ntlc
final_df['hs6'] = final_df['Ntlc'].apply(lambda x: x[:6])
final_df['hs2'] = final_df['Ntlc'].apply(lambda x: x[:2])


# Remove special characters from product descriptions
def clean_string(input_string):
    cleaned_string = re.sub(r'[^a-zA-Z0-9\s]', '', input_string)
    cleaned_string = re.sub('\n', ' ',cleaned_string)
    return cleaned_string

final_df['PdtName'] = final_df['PdtName'].apply(lambda x: clean_string(x.lower()))
final_df['PdtSpecification'] = final_df['PdtSpecification'].apply(lambda x: clean_string(x.lower()))


# Further clean final_df, excluding emply descriptions
final_df = final_df[(final_df['PdtName'] != '') & (final_df['PdtName'] != ' ') & (final_df['PdtSpecification'] != '') & (final_df['PdtSpecification'] != ' ')].reset_index(drop=True)



#######################################################################################################
###########################      2. TRAIN MODEL AND COMPUTE EMBEDDINGS      ###########################
#######################################################################################################
 
# 2.1 Select model/models to test
model = SentenceTransformer("all-MiniLM-L6-v2")


# 2.2 Functions to augment dataset for finetuning, if model needs finetuning

def random_insertion(description, n=1):
    words = description.split()
    for _ in range(n):
        new_word = random.choice(words)
        insert_position = random.randint(0, len(words))
        words.insert(insert_position, new_word)
    new_description = ' '.join(words)
    return new_description

def precompute_different_hs2(train_df, description_feature):
    unique_hs2 = train_df['hs2'].unique()
    different_hs2_dict = {}
    for hs2 in unique_hs2:
        different_hs2_dict[hs2] = train_df[train_df['hs2'] != hs2][description_feature].values
    return different_hs2_dict

def choose_random_hs2(current_hs2, hs2_list):
    choices = [hs2 for hs2 in hs2_list if hs2 != current_hs2]
    return random.choice(choices)

def choose_random_description(hs2_code, descriptions_dict):
    return random.choice(descriptions_dict[hs2_code])

def augment_dataset(train_df, description_feature):
    finetune_df = train_df[[description_feature, 'hs6', 'hs2']].copy()
    
    # Positive pairs
    finetune_df['positive_pair'] = finetune_df[description_feature].apply(random_insertion)
    
    # Negative pairs
    different_hs2_dict = precompute_different_hs2(train_df, description_feature)
    hs2_list = finetune_df['hs2'].unique()
    finetune_df['negative_pair_hs2'] = finetune_df['hs2'].apply(lambda x: choose_random_hs2(x, hs2_list))
    finetune_df['negative_pair'] = finetune_df['hs2'].apply(lambda x: choose_random_description(x, different_hs2_dict))

    finetune_pairs = []
    for i, row in finetune_df.iterrows():
        finetune_pairs.append((row[description_feature], row['positive_pair'], 1))  # Positive pair
        finetune_pairs.append((row[description_feature], row['negative_pair'], 0))  # Negative pair
    
    return finetune_df, finetune_pairs



# 2.3 Function to finetune model, if model needs finetuning

finetune_df, finetune_pairs = augment_dataset(final_df, 'PdtName')

def finetune_model(model, train_df):
    
    #ContrastiveLoss
    
    return xxx




#######################################################################################################
###########################                  3. TEST MODEL                  ###########################
#######################################################################################################

# 3.1 Function for encoding test documents in batches
def encode_batch(batch):
    return model.encode(batch, convert_to_tensor=True)


# 3.2 Function to find k most similar descriptions and corresponding hs6 codes for each document in test set
def most_similar_descriptions(train_df, test_df, train_embed, k):
    
    most_similar_hs6 = []
    
    # split data into batches and perform parallel processing
    batch_size = 64
    batches = [list(test_df['PdtSpecification'])[i:i+batch_size] for i in range(0, len(test_df), batch_size)]
    embeddings = Parallel(n_jobs=-1)(delayed(encode_batch)(batch) for batch in batches)
    test_embed = torch.cat(embeddings)

    for i in test_embed:
        
        # compute similarities
        similarities = model.similarity(test_embed[i], train_embed)

        # get the k most similar documents
        topk_similarities, topk_indices = torch.topk(similarities, k)
        most_similar_docs = train_embed[topk_indices]

        # get the corresponding hs6 code for the k most similar embeddings from train set c
        most_similar_desc.append([train_df['PdtSpecification'][j] for j in topk_indices])
        most_similar_hs6.append([train_df['hs6'][j] for j in topk_indices])

    return most_similar_hs6


# 3.3 Function to compute accuracy of the model
def compute_accuracy(train_df, test_df, train_embed, k):
    
    most_similar_hs6 = most_similar_descriptions(train_df, test_df, train_embed, k)
    test_hs6 = test_df['hs6'].values
    
    # return test documents where hs6 code is found in most similar documents from train set and compute accuracy
    success_array = np.array([test_hs6[i] in most_similar_hs6[i] for i in range(len(test_df))])
    accuracy = np.sum(success_array) / len(test_df)

    return accuracy


#######################################################################################################
###########################         4. FINAL EMBEDDINGS AND QUERIES         ###########################
#######################################################################################################

# Function to encode a single batch
def encode_batch(batch):
    return model.encode(batch, convert_to_tensor=True)

# Split data into batches
batch_size = 64
batches = [list(final_df['PdtSpecification'])[i:i+batch_size] for i in range(0, len(final_df), batch_size)]

# Parallel processing
embeddings = Parallel(n_jobs=-1)(delayed(encode_batch)(batch) for batch in batches)
embed_docs = torch.cat(embeddings)



query = 'beverage drink'
embed_query = model.encode(query)


# Compute similarities
similarities = model.similarity(embed_query, embed_docs)

# Get the k most similar documents
k = 5  
topk_similarities, topk_indices = torch.topk(similarities, k)

# Retrieve the k most similar embed_docs
most_similar_docs = embed_docs[topk_indices]

# If you want to get the corresponding original documents
most_similar_original_docs = [final_df['PdtSpecification'][i] for i in topk_indices]
most_similar_original_hs6 = [final_df['hs6'][i] for i in topk_indices]



final_df.iloc[index_of_highest_similarity,:]