import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as spc
import bamt.Networks as Nets
import bamt.Preprocessors as pp
from varclushi import VarClusHi
from pgmpy.estimators import K2Score
from bamt.utils.MathUtils import get_proximity_matrix
from sklearn import preprocessing
from sklearn.metrics import mutual_info_score
from sklearn.cluster import AgglomerativeClustering
import sys
parentdir = '/Users/jerzykaminski/Documents/GitHub/FEDOT/'
bamtdir = '/Users/jerzykaminski/Documents/GitHub/BAMT/'
sys.path.insert(0, parentdir)
sys.path.insert(0, bamtdir)

def encode_categorical_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features
    :param data: data for encoding
    :return: encoded data
    """
    for col in data.columns:
        if data[col].dtype == 'object':
            le = preprocessing.LabelEncoder()
            data[col] = le.fit_transform(data[col])
    return data


def count(D: list, n_clusters: int):
    model = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='single')
    model = model.fit_predict(D)
    res_dict = {}
    for i, val in enumerate(model):
        if val in res_dict:
            res_dict[val].append(i)
        else:
            res_dict[val] = [i]
    count = 0
    for val in res_dict.values():
        if count < len(val):
            count = len(val)
    return count

def opt_cluster(D: list):
    max_cluster = [count(D, i) for i in range(1, len(D) + 1)]
    dim_space = [max(i, val) for i, val in enumerate(max_cluster)]
    #Optimal number of cluster
    return np.argmin(dim_space) + 1

def mutual_info_clustering(data: pd.DataFrame, cluster_number: int = 2) -> dict:
    """
    Clustering of features based on mutual information
    :param data: data for clustering
    :param cluster_number: number of clusters
    :return: dictionary with clusters
    """
    proximity_matrix = get_proximity_matrix(data, proximity_metric="MI")
    proximity_matrix = 1 - proximity_matrix
    proximity_matrix = np.nan_to_num(proximity_matrix)
    linkage = spc.linkage(proximity_matrix, method='average')
    clusters = spc.fcluster(linkage, cluster_number, criterion='maxclust')
    clusters = {i: list(data.columns[np.where(clusters == i)[0]]) for i in range(1, cluster_number + 1)}
    print(clusters)
    return clusters

def mutual_info_clustering_with_cap(data: pd.DataFrame, cluster_number: int = 2, max_var_number_in_cluster: int = 20) -> dict:
    """
    Clustering of features based on mutual information
    :param data: data for clustering
    :param cluster_number: number of clusters
    :param max_var_number_in_cluster: maximum number of variables in cluster
    :return: dictionary with clusters
    """
    proximity_matrix = get_proximity_matrix(data, proximity_metric="MI")
    proximity_matrix = 1 - proximity_matrix
    proximity_matrix = np.nan_to_num(proximity_matrix)
    linkage = spc.linkage(proximity_matrix, method='average')
    clusters = spc.fcluster(linkage, cluster_number, criterion='maxclust')
    clusters = {i: list(data.columns[np.where(clusters == i)[0]]) for i in range(1, cluster_number + 1)}
    for key, value in clusters.items():
        if len(value) > max_var_number_in_cluster:
            clusters[key] = value[:max_var_number_in_cluster]
    print(clusters)
    return clusters

def varclushi_clustering(data: pd.DataFrame, maxeigval2: int = 1, maxclus: int = 4) -> dict:
    data = encode_categorical_features(data)
    vc = VarClusHi(data, maxeigval2=maxeigval2, maxclus=maxclus)
    vc.varclus()
    clusters_df = vc.rsquare
    clusters_df = clusters_df[['Cluster', 'Variable']]
    clusters = {}
    for i in range(max(clusters_df['Cluster'])+1):
        clusters[i] = list(clusters_df[clusters_df['Cluster'] == i]['Variable'])
    print(clusters)
    return clusters
        


def mutual_info_clustering_with_cap(data: pd.DataFrame, cluster_number: int = 2, max_var_number_in_cluster: int = 20) -> dict:
    """
    Clustering of features based on mutual information
    :param data: data for clustering
    :param cluster_number: number of clusters
    :param max_var_number_in_cluster: maximum number of variables in cluster
    :return: dictionary with clusters
    """
    proximity_matrix = get_proximity_matrix(data, proximity_metric="MI")
    proximity_matrix = 1 - proximity_matrix
    proximity_matrix = np.nan_to_num(proximity_matrix)
    linkage = spc.linkage(proximity_matrix, method='average')
    clusters = spc.fcluster(linkage, cluster_number, criterion='maxclust')
    clusters = {i: list(data.columns[np.where(clusters == i)[0]]) for i in range(1, cluster_number + 1)}
    for key, value in clusters.items():
        if len(value) > max_var_number_in_cluster:
            clusters[key] = value[:max_var_number_in_cluster]
    print(clusters)
    return clusters


def get_complete_directed_graph(nodes: list) -> list:
    """
    Get complete directed graph
    :param nodes: list of nodes
    :return: list of edges
    """
    edges = []
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                edges.append([nodes[i], nodes[j]])
    return edges


def get_blacklist(data: pd.DataFrame,  maxeigval2: int = 1, maxclus: int = 4):
    """
    Get black list of edges
    :param data: data for clustering
    :param number_of_local_structures: number of local structures
    :return: list of edges
    """
    clusters = varclushi_clustering(data, maxeigval2=maxeigval2, maxclus=maxclus)
    black_list = []
    for cluster in clusters.values():
        black_list += get_complete_directed_graph(cluster)
    return black_list, clusters


def get_edges_of_local_structures(data: pd.DataFrame,
                                  datatype: str = 'mixed',
                                  maxeigval2: int = 1,
                                  maxclus: int = 4,
                                  has_logit: bool = True,
                                  use_mixture: bool = True,
                                  parallel_count: int = 4) -> list:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        datatype (str, optional): _description_. Defaults to 'mixed'.
        number_of_local_structures (int, optional): _description_. Defaults to 2.
        has_logit (bool, optional): _description_. Defaults to True.
        use_mixture (bool, optional): _description_. Defaults to True.
        parallel_count (int, optional): _description_. Defaults to 4.

    Returns:
        list: _description_
    """

    from joblib import Parallel, delayed

    blacklist, clusters = get_blacklist(data, maxeigval2=maxeigval2, maxclus=maxclus)

    local_structures_edges = []

    # def wrapper(data: pd.DataFrame,
    #             has_logit: bool = True,
    #             use_mixture: bool = True):
    #     data_cluster = data[cluster]
    #     encoder = preprocessing.LabelEncoder()
    #     discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    #     p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    #     discretized_data, _ = p.apply(data_cluster)
    #     local_discretized_data = discretized_data[cluster]
    #     bn = Nets.HybridBN(has_logit=has_logit, use_mixture=use_mixture)
    #     info = p.info
    #     bn.add_nodes(info)
    #     bn.add_edges(local_discretized_data,  scoring_function=('K2', K2Score))
    #     edges = bn.edges
    #     local_structures_edges += edges

    # Parallel(n_jobs=parallel_count)(delayed(wrapper)(data=data, has_logit=has_logit, use_mixture=use_mixture)(range(n)) for n in clusters.values())


    for cluster in clusters.values():
        data_cluster = data[cluster]
        encoder = preprocessing.LabelEncoder()
        discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
        discretized_data, _ = p.apply(data_cluster)
        local_discretized_data = discretized_data
        if datatype == "mixed":
            bn = Nets.HybridBN(has_logit=has_logit, use_mixture=use_mixture)
        elif datatype == "discrete":
            bn = Nets.DiscreteBN()
        elif datatype == "continuous":
            bn = Nets.ContinuousBN(use_mixture=use_mixture)
        info = p.info
        bn.add_nodes(info)
        bn.add_edges(local_discretized_data,  scoring_function=('K2', K2Score))
        edges = bn.edges
        local_structures_edges += edges

    return local_structures_edges, blacklist
