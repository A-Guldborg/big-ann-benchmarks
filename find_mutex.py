def find_mutex(f1,f2, dataset):
    for point_metadata in dataset:
            filters = point_metadata.nonzero()[1]
            if f1 in filters and f2 in filters:
                    print(str(f1) + " and " + str(f2) + "are not mutex")
                    break

