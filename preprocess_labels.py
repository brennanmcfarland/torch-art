from arctic_flaming_monkey_typhoon.data import retrieval as rt

metadata_path = '/media/guest/Main Storage/HDD Data/CMAopenaccess/data.csv'
out_dir = './preprocessed_data.csv'

metadata, len_metadata, metadata_headers, class_to_index, index_to_class, num_classes = rt.load_metadata(
        metadata_path,
        cols=(0, 18, -4),  # -4 necessary so invalid images are ignored TODO: add parameter for which cols to validate?
        class_cols=(18,)
    )

len_metadata = 31149  # TODO: either the dataset is corrupted/in a different format after this point or the endpoint was down last I tried
metadata = metadata[:len_metadata]

with open(out_dir, 'w+', newline='', encoding="utf8") as out_file:
    for metadatum in metadata:
        out_file.write(metadatum[0] + '.png ' + str(class_to_index[0][metadatum[1]]) + '\n')
