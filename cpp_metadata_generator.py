import benchmark.datasets
from collections import defaultdict


a = benchmark.datasets.YFCC100MDataset()
b = a.get_dataset_metadata()
metadata = defaultdict(list)

i = 0
for point in b:
    for filter_idx in point.indices:
            metadata[filter_idx].append(i)
    i += 1

f = open("yfcc10m.foennindex", "w")
f.write("#include <unordered_map>\n")
f.write("#include <unordered_set>\n\n")
f.write("std::unordered_map<int, std::unordered_set<int>> small_labels =\n")
f.write("{\n")
for k,v in metadata:
  points_as_string = "{" + ", ".join(str(x) for x in v) + "}"
  f.write("{"+str(k)+", std::unordered_set<int>" + points_as_string + "},\n")
f.write("};")
f.close()