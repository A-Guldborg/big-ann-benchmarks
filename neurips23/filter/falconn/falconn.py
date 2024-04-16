from __future__ import print_function

import math
import pdb
import timeit
from collections import defaultdict
from functools import lru_cache

import falconn
import numpy as np
from scipy.sparse import csr_matrix

from benchmark.datasets import DATASETS
from neurips23.filter.base import BaseFilterANN


class FALCONN(BaseFilterANN):
    def __init__(self, metric, index_params):
        self.metric = metric
        self.index_params = index_params

    def filtered_query(self, X, filters, k):
        # self.X -= self.center
        # print("filtered")
        # print(X)
        nq = X.shape[0]
        self.I = -np.ones((nq, k), dtype='int32')
        for (i, query) in enumerate(X):
            # if i % 100 == 0:
            #     print("QUERY")
            #     print(i, query)
            res = self.query_object.find_nearest_neighbor(query, filters[i].indices)
            self.I[i] = res

    def get_results(self):
        # print("FALCONN RESULTS")
        # print(self.I)
        return self.I

    def load_index(self, dataset):
        ds = DATASETS[dataset]()
        self.dataset = ds.get_memmap_dataset()
        # self.dataset /= np.linalg.norm(self.dataset, axis=1).reshape(-1, 1)
        self.dataset_metadata = ds.get_dataset_metadata()

        metadata_dic = defaultdict(lambda: set())
        inverse_metadata = defaultdict(lambda: set())
        # breakpoint()
        # metadata = dict(dataset_metadata.tolil().items())

        for idx, el in dict(self.dataset_metadata.todok().items()).keys():
            if (idx % 100000 == 0):
                print("metadata PROGRESS")
                print(idx)
            metadata_dic[idx].add(el)
            inverse_metadata[el].add(idx)

        center = np.mean(self.dataset, axis=0)
        self.center = center
        # self.dataset -= center


        params_cp = falconn.LSHConstructionParameters()
        params_cp.dimension = len(self.dataset[0])
        params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
        params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
        params_cp.l = 1
        # we set one rotation, since the data is dense enough,
        # for sparse data set it to 2
        params_cp.num_rotations = 1
        params_cp.seed = 5721840
        # we want to use all the available threads to set up
        params_cp.num_setup_threads = 0
        params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
        # we build 18-bit hashes so that each table has
        # 2^18 bins; this is a good choise since 2^18 is of the same
        # order of magnitude as the number of data points
        falconn.compute_number_of_hash_functions(18, params_cp)

        print('Constructing the LSH table')
        t1 = timeit.default_timer()
        table = falconn.LSHIndex(params_cp)


        SMALL_LABEL_THRESHOLD = 0.0001
        filter_size_threshold = int(SMALL_LABEL_THRESHOLD * self.dataset_metadata.shape[0])

        small_labels = {}
        for k,v in inverse_metadata.items():
            if(len(v) <= filter_size_threshold):
                small_labels[k] = v

        print(len(small_labels))
        table.setup(self.dataset, metadata_dic, small_labels)
        t2 = timeit.default_timer()
        print('Done')
        print('Construction time: {}'.format(t2 - t1))

        self.query_object = table.construct_query_object()
        # prolly change but need to figure out how

        return True

    def set_query_arguments(self, query_args):
        if "nprobe" in query_args:
            self.nprobe = query_args["nprobe"]
            self.query_object.set_num_probes(self.nprobe)
            self.qas = query_args
        else:
            self.nprobe = 1

    def __str__(self):
        return f'Falconn({self.qas})'
