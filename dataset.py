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

__version__ = 0.008


class ImagesDataSet:
    def __init__(self,
                 data_images_dir: str = '',
                 data_df_path_filename: str = '',
                 image_size=150,
                 ):
        self.version = "ds_v8"
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

        self.train_datagen = None
        self.val_datagen = None
        self.clean_datagen = None

        self.train_gen = None
        self.val_gen = None
        self.all_gen = None

        self.train_df = pd.DataFrame()
        self.val_df = pd.DataFrame()
        self.train_augmentations_list = [A.Rotate(limit=(-3, 3),
                                                  interpolation=cv2.INTER_LANCZOS4,
                                                  p=0.5),
                                         A.RandomResizedCrop(height=self.image_size,
                                                             width=self.image_size,
                                                             scale=(0.24, 1.12),
                                                             ratio=(0.88, 1.12),
                                                             p=1.0),
                                         A.HorizontalFlip(p=0.5),
                                         A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2),
                                                                    contrast_limit=(-0.2, 0.2),
                                                                    p=0.5),
                                         A.HueSaturationValue(p=0.33),
                                         A.RGBShift(r_shift_limit=15,
                                                    g_shift_limit=15,
                                                    b_shift_limit=15,
                                                    p=0.5),
                                         A.Normalize(p=1.0),
                                         # A.Normalize(mean=(0.5, 0.5, 0.5),
                                         #             std=(0.5, 0.5, 0.5),
                                         #             p=1.0),
                                         ]

        """ 123.68, 116.779, 103.939 and dividing by 58.393, 57.12, 57.375, respectively """

        self.val_augmentations_list = [A.CenterCrop(height=self.image_size,
                                                    width=self.image_size,
                                                    p=1.0),
                                       A.Normalize(p=1.0),
                                       # A.Normalize(mean=(0.5, 0.5, 0.5),
                                       #             std=(0.5, 0.5, 0.5),
                                       #             p=1.0),
                                       ]
        # self.augmentation_kwargs = {'rescale': (1 / 127.5) - 1.0,
        #                             'shear_range': 0.12,
        #                             'zoom_range': 0.12,
        #                             'rotation_range': 3,
        #                             'brightness_range': (0.9, 1.1),
        #                             'horizontal_flip': True,
        #                             }
        # self.rescale_kwargs = {'rescale': (1 / 127.5) - 1.0}
        self.data_split()
        # self.train_df['class_id'].hist()
        pass

    def data_split(self):
        """
        Balanced split to train and val
        Using oversampling method
        """

        self.train_df, self.val_df = train_test_split(self.data_df,
                                                      test_size=self.validation_split,
                                                      random_state=42,
                                                      stratify=self.data_df['class_id'].values
                                                      )
        ncat_bal = self.train_df['class_id'].value_counts().max()
        self.train_df = self.train_df.groupby('class_id', as_index=False).apply(
            lambda g: g.sample(ncat_bal, replace=True, random_state=42)).reset_index(drop=True)

        self.train_df.loc[:, 'split'] = 'train'
        self.val_df.loc[:, 'split'] = 'val'

    def build(self):
        self.train_gen = AugmentedImageDataGenerator(dataframe=self.train_df,
                                                     directory=self.data_images_dir,
                                                     x_col="image_name",
                                                     y_col="class_id",
                                                     num_classes=self.num_classes,
                                                     batch_size=self.batch_size,
                                                     target_size=(self.image_size, self.image_size),
                                                     augmentations_list=self.train_augmentations_list,
                                                     augmentations=True,
                                                     shuffle=True,
                                                     validate_filenames=True,
                                                     cache=True,
                                                     subset='train'
                                                     )

        self.val_gen = AugmentedImageDataGenerator(dataframe=self.val_df,
                                                   directory=self.data_images_dir,
                                                   x_col="image_name",
                                                   y_col="class_id",
                                                   num_classes=self.num_classes,
                                                   batch_size=self.batch_size,
                                                   target_size=(self.image_size, self.image_size),
                                                   augmentations_list=self.val_augmentations_list,
                                                   augmentations=True,
                                                   shuffle=False,
                                                   validate_filenames=True,
                                                   cache=True,
                                                   subset='validation'
                                                   )

        # self.train_datagen = ImageDataGenerator(**self.augmentation_kwargs)
        #
        # self.val_datagen = ImageDataGenerator(**self.rescale_kwargs)

        # self.train_gen = self.train_datagen.flow_from_dataframe(dataframe=self.train_df,
        #                                                         directory=self.data_images_dir,
        #                                                         x_col="image_name",
        #                                                         y_col="class_id",
        #                                                         # subset="training",
        #                                                         validate_filenames=True,
        #                                                         batch_size=self.batch_size,
        #                                                         seed=42,
        #                                                         shuffle=True,
        #                                                         class_mode="categorical",
        #                                                         target_size=(self.image_size, self.image_size)
        #                                                         )

        # self.val_gen = self.val_datagen.flow_from_dataframe(dataframe=self.val_df,
        #                                                     directory=self.data_images_dir,
        #                                                     x_col="image_name",
        #                                                     y_col="class_id",
        #                                                     # subset="validation",
        #                                                     validate_filenames=True,
        #                                                     batch_size=self.batch_size,
        #                                                     seed=42,
        #                                                     shuffle=False,
        #                                                     class_mode="categorical",
        #                                                     target_size=(self.image_size, self.image_size)
        #                                                     )

    def build_check_gen(self, batch_size=32, shuffle=False, augmentation=False, subset='validation'):
        if augmentation:
            kwargs = self.train_augmentations_list
        else:
            kwargs = self.val_augmentations_list

        # self.clean_datagen = ImageDataGenerator(**kwargs)

        # self.all_gen = self.clean_datagen.flow_from_dataframe(dataframe=self.data_df,
        #                                                       directory=self.data_images_dir,
        #                                                       x_col="image_name",
        #                                                       y_col="class_id",
        #                                                       shuffle=shuffle,
        #                                                       batch_size=batch_size,
        #                                                       class_mode="categorical",
        #                                                       target_size=(self.image_size, self.image_size)
        #                                                       )

        self.all_gen = AugmentedImageDataGenerator(dataframe=self.data_df,
                                                   directory=self.data_images_dir,
                                                   x_col="image_name",
                                                   y_col="class_id",
                                                   num_classes=self.num_classes,
                                                   batch_size=batch_size,
                                                   target_size=(self.image_size, self.image_size),
                                                   augmentations_list=kwargs,
                                                   augmentations=True,
                                                   shuffle=shuffle,
                                                   validate_filenames=True,
                                                   cache=True,
                                                   subset=subset
                                                   )
        pass


if __name__ == "__main__":
    print("ok")
