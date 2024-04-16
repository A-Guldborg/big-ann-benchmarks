import benchmark.datasets
from collections import defaultdict
import pdb

a = benchmark.datasets.YFCC100MDataset()
b = a.get_dataset_metadata()
metadata = defaultdict(list)
SMALL_LABEL_THRESHOLD = 0.0001
dataset_size = b.shape[0]
filter_size_threshold = int(SMALL_LABEL_THRESHOLD * dataset_size)
print(filter_size_threshold)

i = 0
for point in b:
    for filter_idx in point.indices:
            metadata[int(filter_idx)].append(i)
    i += 1

print("Done reading metadata")
f = open("yfcc10m.foennindex", "w")
f.write("#include <unordered_map>\n")
f.write("#include <vector>\n\n")
f.write("std::unordered_map<int, std::vector<int>> small_labels =\n")
f.write("{\n")

for k,v in metadata.items():
    if(len(v) <= filter_size_threshold):
        points_as_string = "{" + ", ".join(str(x) for x in v) + "}"
        f.write("{"+str(k)+", " + points_as_string + "},\n")
f.write("};")
f.close()
