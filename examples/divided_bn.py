import pandas as pd
import numpy as np
import bamt.Networks as Nets
import bamt.Preprocessors as pp
from varclushi import VarClusHi
from pgmpy.estimators import K2Score
from sklearn import preprocessing
from sklearn.cluster import KMeans
import sys

parentdir = '/home/jerzy/Documents/GitHub/GitHub/FEDOT'
bamtdir = '/home/jerzy/Documents/GitHub/GitHub/BAMT'
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

def varclushi_clustering(data: pd.DataFrame, maxeigval2: int = 1, maxclus: int = 4) -> dict:
    data = encode_categorical_features(data)
    vc = VarClusHi(data, maxeigval2=maxeigval2, maxclus=maxclus)
    vc.varclus()
    clusters_df = vc.rsquare
    clusters_df = clusters_df[['Cluster', 'Variable']]
    clusters = {}
    for i in range(max(clusters_df['Cluster'])+1):
        clusters[i] = list(clusters_df[clusters_df['Cluster'] == i]['Variable'])
    return clusters


class DividedBN:

    def __init__(self, 
                 data: pd.DataFrame,
                 max_local_structures: int = 8,
                 hidden_nodes_clusters: int = 8):
        """
        :param data: data for clustering
        :param cluster_number: number of clusters
        :param max_var_number_in_cluster: maximum number of variables in cluster
        """
        self.data = data
        self.max_local_structures = max_local_structures
        self.local_structures_nodes = {}
        self.local_structures_edges = {}
        self.hidden_nodes_clusters = hidden_nodes_clusters
        self.hidden_nodes = {}
        self.local_structures_info = {}

    def set_local_structures(self,
                             data,
                             datatype: str = 'mixed',
                             has_logit: bool = True,
                             use_mixture: bool = True,
                             maxeigval2: int = 1,
                             parallel_count: int = 4):
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

        self.local_structures_nodes = varclushi_clustering(data, maxeigval2=maxeigval2, maxclus=self.max_local_structures)

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


        for key in self.local_structures_nodes:
            data_cluster = self.data[self.local_structures_nodes[key]]
            encoder = preprocessing.LabelEncoder()
            discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
            p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
            local_discretized_data, _ = p.apply(data_cluster)
            if datatype == "mixed":
                bn = Nets.HybridBN(has_logit=has_logit, use_mixture=use_mixture)
            elif datatype == "discrete":
                bn = Nets.DiscreteBN()
            elif datatype == "continuous":
                bn = Nets.ContinuousBN()
            info = p.info
            bn.add_nodes(info)
            bn.add_edges(local_discretized_data,  scoring_function=('K2', K2Score))
            edges = bn.edges
            local_structure_info = bn.get_info()
            self.local_structures_edges[key] = edges
            self.local_structures_info[key] = local_structure_info

    def set_hidden_nodes(self, data):

        k_means = KMeans(n_clusters=self.hidden_nodes_clusters)

        for key in self.local_structures_nodes:
            data_cluster = data[self.local_structures_nodes[key]]
            k_means.fit(data_cluster)
            self.hidden_nodes[key] = k_means.fit_predict(data_cluster)

        

