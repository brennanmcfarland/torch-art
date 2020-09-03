from arc23.data import retrieval as rt
import numpy as np

metadata_path = '/media/guest/Main Storage/HDD Data/CMAopenaccess/data.csv'
out_dir = './preprocessed_data.csv'

metadata, len_metadata, metadata_headers, class_to_index, index_to_class, num_classes = rt.load_metadata(
        metadata_path,
        cols=(0, 18, -4),  # -4 necessary so invalid images are ignored TODO: add parameter for which cols to validate?
        class_cols=(18,)
    )

len_metadata = 31149  # TODO: either the dataset is corrupted/in a different format after this point or the endpoint was down last I tried
metadata = metadata[:len_metadata]

metadata = np.array(metadata)
uniques, counts = np.unique(metadata[:, 1], return_counts=True)
count_dict = {u: c for u, c in zip(uniques, counts)}
max_count = max(counts)

with open(out_dir, 'w+', newline='', encoding="utf8") as out_file:
    for metadatum in metadata:
        # omit ambiguous categories
        m = class_to_index[0][metadatum[1]]
        if m == 5 or m == 7 or m == 8 or m == 12 or m == 13 or m == 14 or m == 18 or m == 19 or m == 20 or m == 21 or m == 22 or m == 23 or m == 24 or m == 25 or m == 27 or m == 28 or m == 32 or m == 34 or m == 36 or m == 37 or m == 38 or m == 39 or m == 40 or m == 41 or m == 42 or m == 45 or m == 46 or m == 47 or m == 48 or m == 49 or m == 50 or m == 51 or m == 52 or m == 54 or m == 58 or m == 59 or m == 60 or m == 62 or m == 63 or m == 65:
            pass
        else:
            # equalize the frequency of each class
            for _ in range(max_count // count_dict[metadatum[1]]):
                out_file.write(metadatum[0] + ',' + metadatum[1] + '\n')
