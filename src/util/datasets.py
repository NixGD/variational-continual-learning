import os
import shutil
import urllib.request
import torch.utils.data
from PIL import Image

_NOT_MNIST_URL_L = 'http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz'
_NOT_MNIST_URL_S = 'http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz'
_NM_FILE_L = 'notMNIST_large.tar.gz'
_NM_FILE_S = 'notMNIST_small.tar.gz'
_DATA_DIR = '../data/'
_ASCII_A = 65


class NOTMNIST(torch.utils.data.Dataset):
    def __init__(self, train=True, overwrite=False, transform=None, limit_size=400000):
        """
        Constructs an instance of the NOTMNIST dataset, conforming to the interface
        specified by torch.utils.data.Dataset.

        Args:
            train: true if want train set, false if test set
            overwrite: true if want to re-download and re-unpack existing data files
            transform: transform to apply to images
            limit_size: maximum number of samples to load into dataset
        """
        super().__init__()

        dir_name = _DATA_DIR + 'NOT_MNIST/'
        if not os.path.exists(os.path.dirname(dir_name)):
            os.makedirs(os.path.dirname(dir_name))

        # determine file path based on whether we want large or small version
        file_path = dir_name + _NM_FILE_L if train else dir_name + _NM_FILE_S

        # download large version if required
        if train and not os.path.isfile(file_path):
            print('Downloading notMNIST_large from http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz ...')
            urllib.request.urlretrieve(_NOT_MNIST_URL_L, file_path, reporthook=_download_progress_hook)
        # download small version if needed
        elif not os.path.isfile(file_path):
            print('Downloading notMNIST_small from http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz ...')
            urllib.request.urlretrieve(_NOT_MNIST_URL_S, file_path, reporthook=_download_progress_hook)

        # unpack if not unpacked, or if user wants to overwrite
        unpacked_dir_name = dir_name + 'notMNIST_large/' if train else 'notMNIST_small/'
        if not os.path.exists(unpacked_dir_name) or overwrite:
            print('Unpacking into ' + str(dir_name) + '...')
            shutil.unpack_archive(file_path, dir_name)

        # load dataset from files in folders
        data_root_folder = dir_name + ('notMNIST_large/' if train else 'notMNIST_small')
        image_label_pairs = []
        max_images_per_class = limit_size / 10

        for class_directory in os.listdir(data_root_folder):
            class_label = os.fsdecode(class_directory)
            path_to_class_dir = str(data_root_folder) + '/' + os.fsdecode(class_directory)

            for n, image_file in enumerate(os.listdir(path_to_class_dir), 0):
                if n > max_images_per_class:
                    break

                try:
                    path_to_image_file = path_to_class_dir + '/' + os.fsdecode(image_file)
                    image_label_pairs.append((Image.open(os.fsdecode(path_to_image_file)), ord(class_label) - _ASCII_A))
                except OSError:
                    pass

        # finally, setup actual dataset
        self.transforms = transform
        self.data = image_label_pairs

    def __getitem__(self, index):
        sample = self.data[index]
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.data)


def _download_progress_hook(blocks, block_size, total_size):
    progress = blocks / (total_size / block_size)
    # todo
    pass
