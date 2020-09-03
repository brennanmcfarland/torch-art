from arc23.data import retrieval as rt
import numpy as np

metadata_path = './preprocessed_data.csv'

COL_TYPE = 1
COL_IMG_WEB = 0

metadata, len_metadata, metadata_headers, class_to_index, index_to_class, num_classes = rt.load_metadata(
        metadata_path,
        cols=(COL_IMG_WEB, COL_TYPE),
        class_cols=(COL_TYPE,)
    )

metadata = np.array(metadata)

uniques, counts = np.unique(metadata[:, 1], return_counts=True)

print({ u: c for u, c in zip(uniques, counts)})
