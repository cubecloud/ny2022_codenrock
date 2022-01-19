import os
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
        class_weights = list(class_weight.compute_class_weight('balanced',
                                                               np.unique(self.data_df["class_id"].values),
                                                               self.data_df["class_id"].values
                                                               )
                             )
        self.class_weights = dict(enumerate(class_weights))
        self.validation_split = 0.2
        self.datagen = None
        self.clean_datagen = None

        self.train_gen = None
        self.val_gen = None
        self.all_gen = None
        self.test_ds = None

    def build(self):
        self.datagen = ImageDataGenerator(
                                          # rescale=1. / 255.,
                                          samplewise_center=True,
                                          samplewise_std_normalization=True,
                                          rotation_range=8,
                                          width_shift_range=0.12,
                                          height_shift_range=0.12,
                                          zoom_range=0.1,
                                          brightness_range=(0.9, 1.1),
                                          horizontal_flip=True,
                                          fill_mode='nearest',
                                          validation_split=self.validation_split,
                                          )

        self.train_gen = self.datagen.flow_from_dataframe(dataframe=self.data_df,
                                                          directory=self.data_images_dir,
                                                          x_col="image_name",
                                                          y_col="class_id",
                                                          subset="training",
                                                          validate_filenames=True,
                                                          batch_size=self.batch_size,
                                                          seed=42,
                                                          shuffle=True,
                                                          class_mode="categorical",
                                                          target_size=(self.image_size, self.image_size),
                                                          )

        self.val_gen = self.datagen.flow_from_dataframe(dataframe=self.data_df,
                                                        directory=self.data_images_dir,
                                                        x_col="image_name",
                                                        y_col="class_id",
                                                        subset="validation",
                                                        validate_filenames=True,
                                                        batch_size=self.batch_size,
                                                        seed=42,
                                                        shuffle=True,
                                                        class_mode="categorical",
                                                        target_size=(self.image_size, self.image_size)
                                                        )
        pass

    def build_check_gen(self, batch_size=320):
        self.clean_datagen = ImageDataGenerator(
                                                # rescale=1. / 255.
                                                samplewise_center=True,
                                                samplewise_std_normalization=True,
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

    def create_data_from_gen(self):
        len_data = len(self.all_gen.filenames)
        self.build_check_gen(int(len_data//16))
        for x_data, y_data in self.all_gen:
            continue
        return x_data, y_data

    def build_test_ds(self, image_dir):
        self.test_ds = tf.keras.utils.image_dataset_from_directory(directory=image_dir,
                                                                   labels=None,
                                                                   label_mode='categorical',
                                                                   color_mode="rgb",
                                                                   image_size=(self.image_size, self.image_size)
                                                                   )
        pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def plot_augmentation(datagen, n_rows=2, n_cols=4):
        n_images = n_rows * n_cols
        base_size = 3
        fig_size = (n_cols * base_size, n_rows * base_size)
        fig = plt.figure(figsize=fig_size)
        count = 1
        for r in range(n_images):
            plt.subplot(n_rows, n_cols, count)
            plt.axis('off')
            X, Y = next(datagen)
            X = np.squeeze(X)
            plt.imshow((X.astype('uint8')))
            count += 1
        fig.tight_layout(pad=0.0)
        plt.show()

    dataset = ImagesDataSet(os.path.join(os.getcwd(), "data", "train", "images"),
                            os.path.join(os.getcwd(), "data", "train", "train.csv")
                            )
    dataset.batch_size = 1
    dataset.augmentation = False
    dataset.build()
    plot_augmentation(dataset.train_gen, n_rows=5, n_cols=10)
    plot_augmentation(dataset.val_gen, n_rows=5, n_cols=10)
