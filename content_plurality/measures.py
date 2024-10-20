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
    Computes GVI of embeddings when toxic contents are removed.
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