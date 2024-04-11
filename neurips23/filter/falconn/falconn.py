from __future__ import print_function
from functools import lru_cache
from scipy.sparse import csr_matrix
import numpy as np
import falconn
import timeit
import math
import pdb
from collections import defaultdict

from neurips23.filter.base import BaseFilterANN


class FALCONN(BaseFilterANN):
    def __init__(self, metric, index_params):
        self.metric = metric
        self.index_params = index_params

    def filtered_query(self, query, k):
        print("duuuh")
        return 1

    def get_results(self, query, k):
        return self.filtered_query(query, k)
