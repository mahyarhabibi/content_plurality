import os
import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm


##########################################################
# GVI 
def compute_gvi(X:np.ndarray, scale: float, c_reg=0):
    """
    Computes generalized variance index of input embedding array.
    X: embedding array.
    scale: scaling factor to avoid under/over flow.
    c_reg: regulariation constant.
    """
    cov_matrix = EmpiricalCovariance().fit(X).covariance_
    emp_cov_reg = scale * cov_matrix  + c_reg * np.eye(cov_matrix.shape[0])  # regularization, if needed
    gvi = np.linalg.det(emp_cov_reg)

    return gvi

# drop toxic (and random) tweets by toxicity score 
def gvi_tox_removal(tweet_embd, tweet_ids, df, GVI, scale, metric='tox',
                         tox_scores = np.arange(1, 0.15, -0.05), c_reg=0, SEED=None):
    
    """
    Returns GVI volumes after content moderation at different toxicity thresholds.
    tweet_embd: np ndarray of tweet embeddings.
    tweet_id: np array of tweet ids.
    df: pd.DataFrame contaninig tweet_id, and toxicity scores
    GVI: baseline GVI.
    scale: scaling factor for var-cov matrix of embedding.
    metric: name of the toxicity metric
    tox_scores: np array of toxicity thresholds.
    c_reg: regularization coefficient.
    """

    if SEED is not None:
        np.random.seed(SEED)

    gvi_detox = []
    gvi_random = []
    sample_size = []

    for  score in tqdm(tox_scores):      
        # Filter out tweets with tox scores above score
        not_tox_ids = df[df[metric] <= score]['tweet_id'].values.tolist()  
        X = tweet_embd[np.isin(tweet_ids, not_tox_ids)]
        gvi = compute_gvi(X, scale, c_reg)
        gvi_detox.append(np.log(gvi/GVI))

        # Randomly sample the tweets
        sample_ids = np.random.choice(tweet_ids, size=X.shape[0], replace=False)  
        X = tweet_embd[np.isin(tweet_ids, sample_ids)]
        gvi = compute_gvi(X, scale, c_reg )
        gvi_random.append(np.log(gvi/GVI))

        # sample size of total
        sample_frac = X.shape[0]/tweet_embd.shape[0]
        sample_size.append(sample_frac)

    return gvi_detox, gvi_random, sample_size



#################################################################
# IQR metric
def compute_iqr(X: np.ndarray, q_low = 5, q_high = 95):
    """
    Returns IQR volume of input embedding array between percentile q_low and q_high.
    X: embedding array.
    """
    qh_values = np.percentile(X, q = q_high, axis=0)
    ql_values = np.percentile(X, q = q_low, axis=0)
    
    differences = qh_values - ql_values
    log_diff = np.log(differences)
    
    volume = np.sum(log_diff)
    return volume


def iqr_tox_removal(tweet_embd: np.ndarray, tweet_ids, df, q_low, q_high, metric='tox',
                         tox_scores = np.arange(1, 0.45, -0.05), SEED=None):
    
    """
    Returns IQR volumes after content moderation at different toxicity thresholds.
    tweet_embd: input embedding array.
    tweet_ids: np array of tweet ids.
    df: pd DataFrame of tweet_ids and toxicity socres.
    q_low: lower threshold in compute_iqr.
    q_high: upper threshold in compute_iqr.
    metric: toxicity metric name.
    tox_scores: toxicity removal thresholds.
    """
    
    if SEED is not None:
        np.random.seed(SEED) 
        
    volume_detox = []
    volume_random = []
    sample_size = []
    
    for score in tqdm(tox_scores):   
        
        # remove toxic
        not_tox_ids = df[df[metric] <= score]['tweet_id'].values.tolist()  # Filter out the most toxic tweets
        X = tweet_embd[np.isin(tweet_ids, not_tox_ids)]
        volume = compute_iqr(X, q_low, q_high)
        volume_detox.append(volume)
        
        # remove random
        sample_ids = np.random.choice(tweet_ids, size=X.shape[0], replace=False)
        X = tweet_embd[np.isin(tweet_ids, sample_ids)]
        volume = compute_iqr(X, q_low, q_high)
        volume_random.append(volume)
        
        # sampling ratio
        sample_frac = X.shape[0]/tweet_embd.shape[0]
        sample_size.append(sample_frac)
        
    return volume_detox, volume_random, sample_size

####################################################
# MAD
def mad_tox_removal(embedding, arr_ids, df_tox, tox_scores = np.arange(1, 0.45, -0.05), metric='tox',  SEED=0):
    """
    Returns mean Abs deviation volumes after content moderation at different toxicity thresholds.
    embedding: input embedding array.
    arr_ids: np array of tweet ids.
    df_tox: pd DataFrame of tweet_ids and toxicity socres.
    metric: toxicity metric name.
    tox_scores: toxicity removal thresholds.
    """
    
    np.random.seed(SEED)  
    volume_detox = []
    volume_random = []
    sample_size = []
    
    for score in tqdm(tox_scores):   
        
        # remove toxic
        not_tox_ids = df_tox[df_tox[metric] <= score]['tweet_id'].values.tolist()  # Filter out the most toxic tweets
        X_detox = embedding[np.isin(arr_ids, not_tox_ids)]
        X_detox_mean = np.mean(X_detox, axis=0).reshape(1,-1)
        arr_ad = np.abs(X_detox - X_detox_mean)
        volume = np.mean(arr_ad)
        volume_detox.append(volume)
        
        # remove random
        sample_ids = np.random.choice(arr_ids, size=X_detox.shape[0], replace=False)
        X_sample = embedding[np.isin(arr_ids, sample_ids)]
        X_sample_mean = np.mean(X_sample, axis=0).reshape(1,-1)
        arr_ad = np.abs(X_sample - X_sample_mean)
        volume = np.mean(arr_ad)
        volume_random.append(volume)
        
        # sampling ratio
        sample_frac = X_detox.shape[0]/embedding.shape[0]
        sample_size.append(sample_frac)

    return volume_detox, volume_random, sample_size

#####################################################
# compute pairwise cosine similarity between embeddings
from sklearn.metrics.pairwise import cosine_similarity

def compute_pw_cosim(X, stat='mean'):

    """Returns mean pairwise cosine similarity of the input embedding"""

    # Compute the pairwise cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(X)
    
    # Get the upper triangular values excluding the diagonal 
    # ( avoids double counting and including self similarity)
    upper_tri_indices = np.triu_indices_from(cosine_sim_matrix, k=1)
    upper_tri_values = cosine_sim_matrix[upper_tri_indices]
    
    if stat == 'mean':
        return np.mean(upper_tri_values)
    elif stat == 'median':
        return np.median(upper_tri_values)
    else:
        raise ValueError("Invalid stat specified. Use 'mean' or 'median'.")