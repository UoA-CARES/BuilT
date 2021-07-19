
import os
import json


def dataset_initialize(folder, username, slug, title):
    """ initialize a folder with a a dataset configuration (metadata) file
        Parameters
        ==========
        folder: the folder to initialize the metadata file in
    """
    DATASET_METADATA_FILE = 'dataset-metadata.json'
    
    if not os.path.isdir(folder):
        raise ValueError('Invalid folder: ' + folder)

    ref = username + f'/{slug}'
    licenses = []
    default_license = {'name': 'CC0-1.0'}
    licenses.append(default_license)

    meta_data = {
        'title': title,
        'id': ref,
        'licenses': licenses
    }
    meta_file = os.path.join(folder, DATASET_METADATA_FILE)
    with open(meta_file, 'w') as f:
        json.dump(meta_data, f, indent=2)

    print('Data package template written to: ' + meta_file)
    return meta_file
