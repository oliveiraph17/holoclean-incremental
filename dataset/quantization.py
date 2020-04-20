import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from utils import NULL_REPR


def quantize_km(env, df_raw, num_attr_groups_bins, df_raw_previous=None):
    """
    Kmeans clustering using sklearn
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    Currently do 1D clustering
    :param df_raw: pandas.dataframe
    :param num_attr_groups_bins: list[tuple] where each tuple consists of
    (# of bins, list[str]) where the list[str] is a group of attribues to be
    treated as numerical.
    Groups must be disjoint.

    :return: pandas.dataframe after quantization
    """
    tic = time.time()

    if df_raw_previous is not None:
        df_quantized = pd.concat([df_raw_previous, df_raw]).reset_index(drop=True)
    else:
        df_quantized = df_raw.copy()

    df_quantized.replace('', NULL_REPR, inplace=True)
    df_quantized.fillna(NULL_REPR, inplace=True)

    # Assert groups are disjoint
    num_attrs = [attr for _, group in num_attr_groups_bins for attr in group]
    assert len(set(num_attrs)) == len(num_attrs)

    for bins, attrs in num_attr_groups_bins:
        fil_notnull = (df_quantized[attrs] != NULL_REPR).all(axis=1)

        df_group = df_quantized.loc[fil_notnull, attrs].reset_index(drop=True)
        # Matrix of possibly n-dimension values
        X_attrs = df_group.values.astype(np.float)

        if bins >= np.unique(X_attrs, axis=0).shape[0]:
            # No need to quantize since more bins than unique values.
            continue

        km = KMeans(n_clusters=bins)
        km.fit(X_attrs)

        label_pred = km.labels_
        centroids = km.cluster_centers_

        # Lookup cluster centroids and replace their values.
        df_quantized.loc[fil_notnull, attrs] = np.array([centroids[label_pred[idx]]
            for idx in df_group.index]).astype(str)

    if df_raw_previous is not None:
        df_quantized_previous = df_quantized.head(len(df_raw_previous.index)).reset_index(drop=True)
        df_quantized = df_quantized.tail(len(df_raw.index)).reset_index(drop=True)
    else:
        df_quantized_previous = None

    status = "DONE with quantization"
    toc = time.time()
    return status, toc - tic, df_quantized, df_quantized_previous
