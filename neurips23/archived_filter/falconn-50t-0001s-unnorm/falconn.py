from __future__ import print_function

import math
import pdb
import timeit
from collections import defaultdict
from functools import lru_cache

import multiprocessing

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
        self.X = X.astype(np.float32)
        self.X = self.X - self.center
        # print("filtered")
        # print(X)
        nq = self.X.shape[0]
        self.I = -np.ones((nq, k), dtype='int32')
        for (i, query) in enumerate(self.X):
            if i % 10 == 0:
                print("QUERY", i)
            # query = query.astype(np.float32)
            res = self.query_object.find_nearest_neighbor(query, filters[i].indices)
            self.I[i] = res

    def get_results(self):
        # print("FALCONN RESULTS")
        # print(self.I)
        return self.I

    def load_index(self, dataset):
        ds = DATASETS[dataset]()
        self.dataset = ds.get_dataset()
        # manager = multiprocessing.Manager()
        self.dataset = self.dataset.astype(np.float32)
        # self.dataset = self.dataset / np.linalg.norm(self.dataset, axis=1).reshape(-1, 1)
        self.dataset_metadata = ds.get_dataset_metadata()
        metadata_dic = defaultdict(list)
        inverse_metadata = defaultdict(list)

        # breakpoint()
        # metadata = dict(dataset_metadata.tolil().items())

        metadata_slice = self.dataset_metadata
        start = 0
        worker_id = 0

        # def process_metadata(metadata_slice, start, worker_id, inverse_metadata, metadata_dic):
        i = start
        increments = (metadata_slice.shape[0] // 10)
        print("METADATA SLICE SIZE: ", metadata_slice.shape[0])
        print("METADATA INCREMENTS: ", increments)
        for point in metadata_slice:
                if ((i-start) % increments == 0):
                    print("METADATA PROGRESS FOR WORKER ", worker_id, ": ", ((i-start)/increments) * 10, "%", sep="")
                    # print("Inverse Metadata Size, on worker ", worker_id, ": ", len(inverse_metadata), sep="", end="\n\n")
                for filter_idx in point.indices:
                    int_filter_idx = int(filter_idx)
                    # if int_filter_idx not in inverse_metadata:
                    #     inverse_metadata[int_filter_idx] = list()
                    inverse_metadata[int(filter_idx)].append(i)
                    # if i not in metadata_dic:
                    #     metadata_dic[i] = set()
                    metadata_dic[i].append(int_filter_idx)
                i += 1
        print("METADATA PROGRESS FOR WORKER ", worker_id, ": 100%", sep="")


        # threads = 8
        # workload = dataset_metadata.shape[0] // threads
        # workers = [multiprocessing.Process(target=process_metadata, args=(dataset_metadata[workload * i:workload * (i+1)], workload * i, i, inverse_metadata, metadata_dic))
        #        for i in range(threads)]

        # [worker.start() for worker in workers]
        # [worker.join() for worker in workers]


        # for idx, el in dict(self.dataset_metadata.todok().items()).keys():
        #     metadata_dic[idx].add(el)
        #     inverse_metadata[el].add(idx)

        center = np.mean(self.dataset, axis=0)
        self.center = center
        self.dataset -= center


        params_cp = falconn.LSHConstructionParameters()
        params_cp.dimension = len(self.dataset[0])
        params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
        params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
        params_cp.l = 50
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
        falconn.compute_number_of_hash_functions(24, params_cp)

        print('Constructing the LSH table')
        t1 = timeit.default_timer()
        table = falconn.LSHIndex(params_cp)


        SMALL_LABEL_THRESHOLD = 0.0001
        filter_size_threshold = int(SMALL_LABEL_THRESHOLD * self.dataset_metadata.shape[0])

        small_labels = {}
        # print(len(self.inverse_metadata))
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
        return f'Falconn-50t-0.0001s({self.qas})'
