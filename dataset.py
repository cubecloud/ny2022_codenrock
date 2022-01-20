import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

__version__ = 0.005


class ImagesDataSet:
    def __init__(self,
                 data_images_dir: str = '',
                 data_df_path_filename: str = '',
                 image_size=150,
                 ):
        self.version = "v3"
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
        self.test_ds = None
        self.train_df = pd.DataFrame()
        self.val_df = pd.DataFrame()

    def build(self):
        self.train_df, self.val_df = train_test_split(self.data_df,
                                                      test_size=self.validation_split,
                                                      random_state=42,
                                                      stratify=self.data_df['class_id'].values
                                                      )

        self.train_df.loc[:, 'split'] = 'train'
        self.val_df.loc[:, 'split'] = 'val'

        self.train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.1,
            zoom_range=0.1,
            rotation_range=2,
            brightness_range=(0.9, 1.1),
            horizontal_flip=True,
        )

        self.val_datagen = ImageDataGenerator(rescale=1. / 255)

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

    def build_check_gen(self, batch_size=32):
        self.clean_datagen = ImageDataGenerator(
            rescale=1. / 255.
            # samplewise_center=True,
            # samplewise_std_normalization=True,
        )
        self.all_gen = self.clean_datagen.flow_from_dataframe(dataframe=self.data_df,
                                                              directory=self.data_images_dir,
                                                              x_col="image_name",
                                                              y_col="class_id",
                                                              shuffle=False,
                                                              batch_size=batch_size,
                                                              class_mode="categorical",
                                                              target_size=(self.image_size, self.image_size)
                                                              )

    def build_test_ds(self, image_dir):
        self.test_ds = tf.keras.utils.image_dataset_from_directory(directory=image_dir,
                                                                   labels=None,
                                                                   label_mode='categorical',
                                                                   color_mode="rgb",
                                                                   image_size=(self.image_size, self.image_size)
                                                                   )
        pass


if __name__ == "__main__":
    print("ok")
