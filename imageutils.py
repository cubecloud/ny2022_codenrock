import os
import random
import numpy as np
import pandas as pd
from typing import Tuple
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt
import albumentations as A
import cv2
# import copy

import warnings

warnings.filterwarnings('ignore')

__version__ = 0.003


class AugmentedImageDataGenerator(Sequence):
    """
    Custom image data generator.
    Behaves like ImageDataGenerator, but using albumentation.
    Currently, support only categorical
    """

    def __init__(self,
                 dataframe: pd.DataFrame,
                 directory: str,
                 x_col: str,
                 num_classes: int,
                 batch_size: int,
                 target_size: Tuple,
                 y_col: str = None,
                 augmentations_list: object = None,
                 augmentations: bool = True,
                 shuffle: bool = True,
                 validate_filenames: bool = False,
                 cache: bool = True,
                 subset: str = 'train',
                 ):
        # super().__init__(
        #     preprocessing_function=self.augment_pairs,
        #     **kwargs)

        self.df = dataframe
        self.directory = directory
        self.x_col = x_col
        self.y_col = y_col
        self.augmentations_list = augmentations_list
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.shuffle = shuffle
        self.target_size = target_size
        self.img_height = self.target_size[0]
        self.img_width = self.target_size[1]
        self.validate_filenames = validate_filenames
        self.subset = subset
        self.__check_directory()
        self.path_filenames_list: list = []
        self.classes_names_list: list = []
        self.x_len: int = 0
        self.__check_filenames()
        self.len: int = 0
        self.len_calc()

        self.x_cache: list = []

        self.num_classes = num_classes

        if self.y_col:
            self.y_set = tf.keras.utils.to_categorical(y=self.classes_names_list,
                                                       num_classes=self.num_classes,
                                                       )
        else:
            self.y_set = None

        self.indexes = np.arange(self.x_len)

        if cache:
            self.__prefetch()

        self.batch_indexes = self.indexes[0:self.batch_size]
        self.default_train_augmentations_list = [A.Rotate(limit=(3, 3),
                                                          p=0.5),
                                                 A.RandomResizedCrop(height=self.img_height,
                                                                     width=self.img_width,
                                                                     scale=(0.08, 1.0),
                                                                     ratio=(0.88, 1.12),
                                                                     p=1.0),
                                                 A.HorizontalFlip(p=0.5),
                                                 A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                                                            contrast_limit=(-0.1, 0.1),
                                                                            p=0.5),
                                                 A.HueSaturationValue(p=0.5),
                                                 A.RGBShift(r_shift_limit=15,
                                                            g_shift_limit=15,
                                                            b_shift_limit=15,
                                                            p=0.3),
                                                 A.Normalize(p=1.0),
                                                 # A.Normalize(mean=(0.5, 0.5, 0.5),
                                                 #             std=(0.5, 0.5, 0.5)
                                                 #             ),
                                                 # A.Normalize(mean=(0.485, 0.456, 0.406),
                                                 #             std=(0.229*2, 0.224*2, 0.225*2)
                                                 #             ),
                                                 ]

        self.default_val_augmentations_list = [A.CenterCrop(height=self.img_height,
                                                            width=self.img_width,
                                                            ),
                                               A.Normalize(p=1.0),
                                               # A.Normalize(mean=(0.5, 0.5, 0.5),
                                               #             std=(0.5, 0.5, 0.5)
                                               #             ),
                                               # A.Normalize(mean=(0.485, 0.456, 0.406),
                                               #             std=(0.229, 0.224, 0.225)
                                               #             ),
                                               ]

        if self.augmentations:
            if not self.augmentations_list:
                print("Warning: augmentaions list is empty! Used default augmentations list.")
                self.augmentations_list = self.default_train_augmentations_list

        self.a_transform = A.Compose(self.augmentations_list, p=1)

    def __check_directory(self):
        msg = f"Error: {self.directory} does not exists"
        assert os.path.exists(self.directory), msg

    def __check_filenames(self):
        count = 0
        if self.subset != 'test' and self.y_col:
            for filename, class_name in zip(self.df[self.x_col], self.df[self.y_col]):
                path_filename = os.path.join(self.directory, filename)
                if self.validate_filenames:
                    if os.path.isfile(path_filename):
                        self.path_filenames_list.append(path_filename)
                        self.classes_names_list.append(class_name)
                    else:
                        count += 1
                else:
                    self.path_filenames_list.append(path_filename)
                    self.classes_names_list.append(class_name)
        else:
            for filename in self.df[self.x_col]:
                path_filename = os.path.join(self.directory, filename)
                if self.validate_filenames:
                    if os.path.isfile(path_filename):
                        self.path_filenames_list.append(path_filename)
                    else:
                        count += 1
                else:
                    self.path_filenames_list.append(path_filename)

        if self.validate_filenames:
            msg = f"{self.subset} subset: validated {self.df.shape[0]}/{len(self.path_filenames_list)} " \
                  f"records and {len(set(self.path_filenames_list))} files in {self.directory}"
            print(msg)
        self.x_len = len(self.path_filenames_list)
        pass

    def __get_image_from_file(self, path_filename):
        image = cv2.imread(path_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __get_image(self, idx):
        return self.__get_image_from_file(self.path_filenames_list[idx])

    def __prefetch(self):
        if self.subset == 'train':
            prefetch_width = self.img_width + self.img_width // 10
            prefetch_height = self.img_height + self.img_height // 10
        elif self.subset == 'validation' or self.subset == 'test':
            prefetch_width = self.img_width
            prefetch_height = self.img_height
        else:
            msg = f'Error: unknown subset type {self.subset}'
            assert self.subset == 'train' or self.subset == 'validation' or self.subset == 'test', msg

        for path_filename in self.path_filenames_list:
            image = self.__get_image_from_file(path_filename)

            transform = A.Compose(
                [A.SmallestMaxSize(max_size=prefetch_width if image.shape[0] < image.shape[1] else prefetch_height,
                                   interpolation=cv2.INTER_LANCZOS4,
                                   ),
                 ]
                )

            if image.shape[0] <= prefetch_height or image.shape[1] <= prefetch_width:
                augmented = transform(image=image)
                image_augm = augmented['image']
                self.x_cache.append(image_augm)
            else:
                self.x_cache.append(image)
        self.prefetch_status = True

    def check(self, index):
        return self.__getitem__(index)

    def len_calc(self):
        batches = self.x_len / self.batch_size
        self.len = int(max(int(batches), int(batches) if batches.is_integer() else int(batches) + 1))

    def __len__(self):
        """ Denotes the number of batches per epoch """
        return self.len

    def __getitem__(self, index):
        data_index_min = int(index * self.batch_size)
        data_index_max = int(min((index + 1) * self.batch_size, self.x_len))

        indexes = self.indexes[data_index_min:data_index_max]
        this_batch_size = len(indexes)  # The last batch can be smaller than the others

        X_batch = np.empty((this_batch_size, self.img_height, self.img_width, 3), dtype=np.float32)
        y_batch = np.empty((this_batch_size, self.num_classes), dtype=np.uint8)

        for i, sample_index in enumerate(indexes):
            if self.x_cache:
                X_sample = self.x_cache[sample_index]
            else:
                X_sample = self.__get_image(sample_index)

            if self.augmentations:
                augmented = self.a_transform(image=X_sample)
                image_augm = augmented['image']
                X_batch[i, ...] = image_augm
                if self.y_set is not None:
                    y_sample = self.y_set[sample_index]
                    y_batch[i, ...] = y_sample
            else:
                X_batch[i, ...] = X_sample
                if self.y_set is not None:
                    y_sample = self.y_set[sample_index]
                    y_batch[i, ...] = y_sample
        self.batch_indexes = indexes

        if self.y_set is not None:
            return X_batch, y_batch
        else:
            return X_batch

    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        self.indexes = np.arange(self.x_len)
        if self.shuffle:
            random.seed(42)
            np.random.shuffle(self.indexes)


if __name__ == "__main__":
    home_dir = os.getcwd()
    base_dir = os.path.join(home_dir, 'data')
    train_dir = os.path.join(base_dir, 'train')
    weight_dir = os.path.join(base_dir, 'weight')
    test_dir = os.path.join(base_dir, 'test')

    data_df = pd.read_csv(os.path.join(base_dir, "train.csv"),
                          delimiter="\t",
                          dtype={'image_name': str,
                                 'class_id': str
                                 }
                          )

    test_datagen = AugmentedImageDataGenerator(dataframe=data_df,
                                               directory=train_dir,
                                               x_col="image_name",
                                               y_col="class_id",
                                               num_classes=3,
                                               batch_size=15,
                                               target_size=(224, 224),
                                               augmentations_list=None,
                                               augmentations=True,
                                               shuffle=True,
                                               validate_filenames=False,
                                               cache=True,
                                               subset='train'
                                               )


    def display_image_grid(images_filepaths, predicted_labels=(), cols=5):
        rows = len(images_filepaths) // cols
        figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
        for i, image_filepath in enumerate(images_filepaths):
            image = cv2.imread(image_filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            true_label = os.path.normpath(image_filepath).split(os.sep)[-2]
            predicted_label = predicted_labels[i] if predicted_labels else true_label
            color = "green" if true_label == predicted_label else "red"
            ax.ravel()[i].imshow(image.astype('uint8'))
            ax.ravel()[i].set_title(predicted_label, color=color)
            ax.ravel()[i].set_axis_off()
        plt.tight_layout()
        plt.show()


    def visualize_augmentations(datagen, idx=0, samples=10, cols=5):
        # dataset = copy.deepcopy(datagen)
        # dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
        batch_x, batch_y = datagen.__getitem__(idx)
        rows = samples // cols
        figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
        for i in range(len(batch_x)):
            image = batch_x[i]
            ax.ravel()[i].imshow(image.astype('uint8'))
            ax.ravel()[i].set_title(np.argmax(batch_y[i]), color="green")
            ax.ravel()[i].set_axis_off()
        plt.tight_layout()
        plt.show()


    display_image_grid(test_datagen.path_filenames_list[:15], test_datagen.classes_names_list[:15])
    visualize_augmentations(test_datagen, samples=15)

    test_datagen = AugmentedImageDataGenerator(dataframe=data_df,
                                               directory=test_dir,
                                               x_col="image_name",
                                               y_col=None,
                                               num_classes=3,
                                               batch_size=630,
                                               target_size=(224, 224),
                                               augmentations_list=None,
                                               augmentations=True,
                                               shuffle=False,
                                               validate_filenames=True,
                                               cache=True,
                                               subset='test'
                                               )

    batch_x, batch_y = test_datagen.__getitem__(2)
    print(np.min(batch_x), np.mean(batch_x), np.max(batch_x))

    print("Ok")
