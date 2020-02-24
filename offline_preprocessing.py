import random
import arc23.data.retrieval as rt
import arc23.data.handling as dt
import arc23.transforms.image_transforms as it
from arc23.functional.core import pipe


metadata_path = '/media/guest/Main Storage/HDD Data/CMAopenaccess/data.csv'
data_in_dir = '/media/guest/Main Storage/HDD Data/CMAopenaccess/images/'
data_out_dir = '/media/guest/Main Storage/HDD Data/CMAopenaccess/preprocessed_images/'


# TODO: this is copied from app.py, move it somewhere else (get_label too)
def get_from_metadata():
    get = rt.get_img_from_file_or_url(img_format='JPEG')

    def _apply(metadatum):
        filepath = data_in_dir + metadatum[0] + '.jpg'
        url = metadatum[-1]
        return get(filepath, url)
    return _apply


def get_label(metadatum):
    return metadatum[1]



def run():
    metadata, len_metadata, metadata_headers, class_to_index, index_to_class, num_classes = rt.load_metadata(
        metadata_path,
        cols=(COL_ID, COL_TYPE, COL_IMG_WEB),
        class_cols=(COL_TYPE,)
    )
    len_metadata = 31149  # TODO: either the dataset is corrupted/in a different format after this point or the endpoint was down last I tried
    metadata = metadata[:len_metadata]

    # TODO: abstract into some sort of pipeline?
    # TODO: use DALI
    for metadatum in metadata:
        img = pipe(get_from_metadata(), it.random_fit_to((256, 256)))(metadatum)
        filepath = data_out_dir + metadatum[0] + '.png'
        img.save(filepath, img_format='PNG')
        print('preprocessed ', metadatum[0])


if __name__ == '__main__':
    COL_ID = 0
    COL_CREATION_DATE = 9
    COL_TYPE = 18
    COL_IMG_WEB = -4  # lower resolution makes for smaller dataset/faster training
    run()
