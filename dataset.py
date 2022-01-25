import numpy as np
import pandas as pd
from sklearn.utils import class_weight, shuffle
from sklearn.model_selection import train_test_split
from keras.applications import imagenet_utils
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import albumentations as A
from imageutils import AugmentedImageDataGenerator
import cv2

__version__ = 0.018


class ImagesDataSet:
    version = "ds_v18"

    def __init__(self,
                 data_images_dir: str = '',
                 data_df_path_filename: str = '',
                 image_size=150,
                 color_mode='RGB',
                 balancing=False,
                 ):
        self.version = self.__class__.version
        if color_mode != 'RGB':
            self.version = f'{self.version}_{color_mode}'

        self.image_size = image_size
        assert data_images_dir, "Error: set the train directory!"
        self.data_images_dir = data_images_dir
        assert data_df_path_filename, "Error: set the train.csv file!"
        self.data_df = pd.read_csv(data_df_path_filename,
                                   delimiter="\t",
                                   dtype={'image_name': str,
                                          'class_id': str
                                          }
                                   )
        self.batch_size = 32
        self.num_classes = self.data_df["class_id"].nunique()
        cl_weights = list(class_weight.compute_class_weight('balanced',
                                                            np.unique(self.data_df["class_id"].values),
                                                            self.data_df["class_id"].values)
                          )
        self.class_weights = dict(enumerate(cl_weights))
        self.validation_split = 0.2
        self.color_mode = color_mode
        self.balancing = balancing

        self.train_datagen = None
        self.val_datagen = None
        self.clean_datagen = None

        self.train_gen = None
        self.val_gen = None
        self.all_gen = None

        self.train_df = pd.DataFrame()
        self.val_df = pd.DataFrame()
        # self.train_augmentations_list = [A.Rotate(limit=(-3, 3),
        #                                           interpolation=cv2.INTER_LANCZOS4,
        #                                           p=0.5),
        #                                  A.RandomResizedCrop(height=self.image_size,
        #                                                      width=self.image_size,
        #                                                      scale=(0.08, 1.0),
        #                                                      ratio=(0.88, 1.0),
        #                                                      p=1.0),
        #                                  A.HorizontalFlip(p=0.5),
        #                                  A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
        #                                                             contrast_limit=(-0.1, 0.1),
        #                                                             p=0.5),
        #                                  A.HueSaturationValue(p=0.5),
        #                                  A.RGBShift(r_shift_limit=15,
        #                                             g_shift_limit=15,
        #                                             b_shift_limit=15,
        #                                             p=0.5),
        #                                  A.FancyPCA(p=1.0),
        #                                  # A.Normalize(p=1.0),
        #                                  A.Normalize(mean=(0.5, 0.5, 0.5),
        #                                              std=(0.5, 0.5, 0.5),
        #                                              p=1.0),
        #                                  ]

        """ 123.68, 116.779, 103.939 and dividing by 58.393, 57.12, 57.375, respectively """

        # self.val_augmentations_list = [A.CenterCrop(height=self.image_size,
        #                                             width=self.image_size,
        #                                             p=1.0),
        #                                # A.Normalize(p=1.0),
        #                                A.Normalize(mean=(0.5, 0.5, 0.5),
        #                                            std=(0.5, 0.5, 0.5),
        #                                            p=1.0),
        #                                ]
        self.augmentation_kwargs = {'rescale': (1 / 127.5) - 1.0,
                                    'shear_range': 0.12,
                                    'zoom_range': 0.12,
                                    'rotation_range': 3,
                                    'brightness_range': (0.9, 1.1),
                                    'horizontal_flip': True,
                                    }
        self.rescale_kwargs = {'rescale': (1 / 127.5) - 1.0}
        pass

    def data_split(self, balancing=False):
        """
        Balanced split to train and val
        Using oversampling method
        """
        self.train_df, self.val_df = train_test_split(self.data_df,
                                                      test_size=self.validation_split,
                                                      random_state=42,
                                                      stratify=self.data_df['class_id'].values
                                                      )

        if balancing:
            train_classes, train_count = np.unique(self.train_df["class_id"], return_counts=True)
            val_classes, val_count = np.unique(self.val_df["class_id"], return_counts=True)
            train_files_count = len(np.unique(self.train_df["image_name"]))
            val_files_count = len(np.unique(self.val_df["image_name"]))
            msg = f'Dataframe split _before_ balancing: \n' \
                  f'train: {self.train_df.shape[0]} records. Classes: {train_classes}, ' \
                  f'classes_count: {train_count}, files_count {train_files_count} \n' \
                  f'val: {self.val_df.shape[0]} records. Classes: {val_classes}, ' \
                  f'classes_count: {val_count}, files_count {val_files_count} \n'
            print(msg)

            ncat_bal = np.max(train_count)
            new_df = self.train_df.copy()
            for class_name, class_count in zip(train_classes, train_count):
                if class_count == ncat_bal:
                    continue
                mask = self.train_df['class_id'] == class_name
                bal_def = ncat_bal - class_count
                frac_def = bal_def / class_count
                if int(frac_def) == 0:
                    temp_df = self.train_df.loc[mask].sample(n=bal_def, random_state=24)
                else:
                    for i in range(int(frac_def)):
                        temp_df = self.train_df.loc[mask]
                        new_df = pd.concat([temp_df, new_df])
                    temp_df = self.train_df.loc[mask].sample(n=bal_def - (int(frac_def) * class_count), replace=True,
                                                             random_state=24)
                new_df = pd.concat([temp_df, new_df])

            self.train_df = new_df
            self.train_df.reset_index(inplace=True, drop=True)
            train_classes, train_count = np.unique(self.train_df["class_id"], return_counts=True)
            val_classes, val_count = np.unique(self.val_df["class_id"], return_counts=True)
            train_files_count = len(np.unique(self.train_df["image_name"]))
            val_files_count = len(np.unique(self.val_df["image_name"]))

            msg = f'Dataframe split _after_ balancing: \n' \
                  f'train: {self.train_df.shape[0]} records. Classes: {train_classes}, ' \
                  f'classes_count: {train_count}, files_count {train_files_count} \n' \
                  f'val: {self.val_df.shape[0]} records. Classes: {val_classes}, ' \
                  f'classes_count: {val_count}, files_count {val_files_count} \n'
            print(msg)

        self.train_df.loc[:, 'split'] = 'train'
        self.val_df.loc[:, 'split'] = 'val'
        pass

    def build(self):
        self.data_split(balancing=self.balancing)
        # self.train_gen = AugmentedImageDataGenerator(dataframe=self.train_df,
        #                                              directory=self.data_images_dir,
        #                                              x_col="image_name",
        #                                              y_col="class_id",
        #                                              num_classes=self.num_classes,
        #                                              batch_size=self.batch_size,
        #                                              target_size=(self.image_size, self.image_size),
        #                                              augmentations_list=self.train_augmentations_list,
        #                                              augmentations=True,
        #                                              shuffle=True,
        #                                              validate_filenames=True,
        #                                              cache=True,
        #                                              subset='train',
        #                                              color_mode=self.color_mode
        #                                              )
        #
        # self.val_gen = AugmentedImageDataGenerator(dataframe=self.val_df,
        #                                            directory=self.data_images_dir,
        #                                            x_col="image_name",
        #                                            y_col="class_id",
        #                                            num_classes=self.num_classes,
        #                                            batch_size=self.batch_size,
        #                                            target_size=(self.image_size, self.image_size),
        #                                            augmentations_list=self.val_augmentations_list,
        #                                            augmentations=True,
        #                                            shuffle=False,
        #                                            validate_filenames=True,
        #                                            cache=True,
        #                                            subset='validation',
        #                                            color_mode=self.color_mode
        #                                            )

        self.train_datagen = ImageDataGenerator(**self.augmentation_kwargs)

        self.val_datagen = ImageDataGenerator(**self.rescale_kwargs)

        self.train_gen = self.train_datagen.flow_from_dataframe(dataframe=self.train_df,
                                                                directory=self.data_images_dir,
                                                                x_col="image_name",
                                                                y_col="class_id",
                                                                # subset="training",
                                                                validate_filenames=True,
                                                                batch_size=self.batch_size,
                                                                seed=42,
                                                                shuffle=True,
                                                                class_mode="categorical",
                                                                target_size=(self.image_size, self.image_size)
                                                                )

        self.val_gen = self.val_datagen.flow_from_dataframe(dataframe=self.val_df,
                                                            directory=self.data_images_dir,
                                                            x_col="image_name",
                                                            y_col="class_id",
                                                            # subset="validation",
                                                            validate_filenames=True,
                                                            batch_size=self.batch_size,
                                                            seed=42,
                                                            shuffle=False,
                                                            class_mode="categorical",
                                                            target_size=(self.image_size, self.image_size)
                                                            )

    def build_check_gen(self, batch_size=32, shuffle=False, augmentation=False, subset='validation'):
        # if augmentation:
        #     kwargs = self.train_augmentations_list
        # else:
        #     kwargs = self.val_augmentations_list

        if augmentation:
            kwargs = {'rescale': (1 / 127.5) - 1.0,
                                    'shear_range': 0.12,
                                    'zoom_range': 0.12,
                                    'rotation_range': 3,
                                    'brightness_range': (0.9, 1.1),
                                    'horizontal_flip': True,
                                    }
        else:
            kwargs = {'rescale': (1 / 127.5) - 1.0}


        self.clean_datagen = ImageDataGenerator(**kwargs)

        self.all_gen = self.clean_datagen.flow_from_dataframe(dataframe=self.data_df,
                                                              directory=self.data_images_dir,
                                                              x_col="image_name",
                                                              y_col="class_id",
                                                              shuffle=shuffle,
                                                              batch_size=batch_size,
                                                              class_mode="categorical",
                                                              target_size=(self.image_size, self.image_size)
                                                              )

        # self.all_gen = AugmentedImageDataGenerator(dataframe=self.data_df,
        #                                            directory=self.data_images_dir,
        #                                            x_col="image_name",
        #                                            y_col="class_id",
        #                                            num_classes=self.num_classes,
        #                                            batch_size=batch_size,
        #                                            target_size=(self.image_size, self.image_size),
        #                                            augmentations_list=kwargs,
        #                                            augmentations=True,
        #                                            shuffle=shuffle,
        #                                            validate_filenames=True,
        #                                            cache=True,
        #                                            subset=subset,
        #                                            color_mode=self.color_mode
        #                                            )
        # pass


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    import random


    def plot_augmentation(datagen, n_rows=2, n_cols=4):
        n_images = n_rows * n_cols
        base_size = 3
        fig_size = (n_cols * base_size, n_rows * base_size)
        fig = plt.figure(figsize=fig_size)
        count = 1
        images_indices = random.sample(range(datagen.__len__()), k=n_images)
        for ix in (images_indices):
            plt.subplot(n_rows, n_cols, count)
            plt.axis('off')
            X, Y = datagen.__getitem__(ix)
            X = np.squeeze(X)
            plt.imshow((X.astype('uint8')))
            count += 1
        fig.tight_layout(pad=0.0)
        plt.show()


    dataset = ImagesDataSet(os.path.join(os.getcwd(), "data", "train"),
                            os.path.join(os.getcwd(), "data", "train.csv")
                            )
    dataset.validation_split = 0.1
    dataset.batch_size = 1
    dataset.build()
    plot_augmentation(dataset.train_gen, n_rows=5, n_cols=10)
    plot_augmentation(dataset.val_gen, n_rows=5, n_cols=10)
    print("ok")
