import os
from typing import Tuple
import pytz
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from models import sepconv2d, resnet50v2_classification_model
from dataset import ImagesDataSet
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

__version__ = 0.003

home_dir = os.getcwd()
base_dir = os.path.join(home_dir, 'data')
train_dir = os.path.join(base_dir, 'train')
weight_dir = os.path.join(base_dir, 'weight')
test_dir = os.path.join(base_dir, 'test')


class TrainNN:
    def __init__(self,
                 dataset: ImagesDataSet,
                 ):
        self.dataset = dataset
        self.y_Pred = None
        self.experiment_name = f"{self.dataset.version}"
        self.history = None
        self.epochs = 15

        """ Use it only if not using TimeSeries Generator"""
        self.batch_size = None
        self.monitor = "val_categorical_accuracy"
        self.loss = "categorical_crossentropy"
        self.metric = "categorical_accuracy"
        self.path_filename: str = ''
        self.model_compiled = False
        self.es_patience = 15
        self.rlrs_patience = 8
        self.keras_model, self.net_name = sepconv2d(input_shape=(self.dataset.image_size,
                                                                 self.dataset.image_size) + (3,),
                                                    num_classes=3)
        self.learning_rate = 3e-5
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.class_weights = self.dataset.class_weights
        # self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, nesterov=True, momentum=0.9)

    def compile(self):
        self.path_filename = os.path.join(weight_dir, f"{self.experiment_name}_{self.net_name}")
        self.keras_model.summary()
        self.keras_model.compile(optimizer=self.optimizer,
                                 loss=self.loss,
                                 metrics=[self.metric],
                                 )
        self.model_compiled = True
        pass

    def train(self):
        if not self.model_compiled:
            self.compile()
        chkp = ModelCheckpoint(os.path.join(weight_dir, f"{self.experiment_name}_{self.net_name}.h5"),
                               mode='min',
                               monitor=self.monitor,
                               save_best_only=True,
                               )
        rlrs = ReduceLROnPlateau(monitor=self.monitor, factor=0.2, patience=self.rlrs_patience, min_lr=1e-07)
        es = EarlyStopping(patience=self.es_patience, monitor=self.monitor, restore_best_weights=True)
        callbacks = [rlrs, chkp, es]
        path_filename = f"{self.path_filename}_NN.png"

        tf.keras.utils.plot_model(self.keras_model,
                                  to_file=path_filename,
                                  show_shapes=True,
                                  show_layer_names=True,
                                  expand_nested=True,
                                  dpi=96,
                                  )
        self.history = self.keras_model.fit(self.dataset.train_gen,
                                            validation_data=self.dataset.val_gen,
                                            epochs=self.epochs,
                                            verbose=1,
                                            callbacks=callbacks,
                                            class_weight=self.class_weights
                                            )
        self.model_compiled = True
        pass

    def load_best_weights(self):
        path_filename = f"{self.path_filename}.h5"
        self.keras_model.load_weights(path_filename)
        pass

    def get_predict(self, x_Data):
        if not self.model_compiled:
            self.compile()
        self.load_best_weights()
        self.y_Pred = self.keras_model.predict(x_Data)
        return self.y_Pred

    def evaluate(self, Test: Tuple):
        if not self.model_compiled:
            self.compile()
        self.load_best_weights()
        self.keras_model.evaluate(Test)
        pass

    def figshow_base(self):
        sub_plots = 1
        if self.monitor != "val_loss" and ("accuracy" not in self.metric):
            sub_plots = 2
        fig = plt.figure(figsize=(24, 10*sub_plots))
        sns.set_style("white")
        ax1 = fig.add_subplot(1, sub_plots, 1)
        ax1.set_axisbelow(True)
        ax1.minorticks_on()
        # Turn on the minor TICKS, which are required for the minor GRID
        # Customize the major grid
        ax1.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        # Customize the minor grid
        ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        N = np.arange(0, len(self.history.history["loss"]))
        plt.plot(N, self.history.history["loss"], label="loss")
        if 'val_loss' in self.history.history:
            plt.plot(N, self.history.history["val_loss"], label="val_loss")
        if 'dice_coef' in self.history.history:
            plt.plot(N, self.history.history["dice_coef"], label="dice_coef")
        if 'val_dice_coef' in self.history.history:
            plt.plot(N, self.history.history["val_dice_coef"], label="val_dice_coef")
        if 'mae' in self.history.history:
            plt.plot(N, self.history.history["mae"], label="mae")
        if 'accuracy' in self.history.history:
            plt.plot(N, self.history.history["accuracy"], label="accuracy")
        if 'val_accuracy' in self.history.history:
            plt.plot(N, self.history.history["val_accuracy"], label="val_accuracy")
        if 'categorical_accuracy' in self.history.history:
            plt.plot(N, self.history.history["categorical_accuracy"], label="categorical_accuracy")
        if 'val_categorical_accuracy' in self.history.history:
            plt.plot(N, self.history.history["val_categorical_accuracy"], label="val_categorical_accuracy")
        plt.title(f"Training Loss and Metric")
        plt.legend()
        if sub_plots == 2:
            ax2 = fig.add_subplot(1, sub_plots, sub_plots)
            ax2.set_axisbelow(True)
            ax2.minorticks_on()
            # Turn on the minor TICKS, which are required for the minor GRID
            # Customize the major grid
            ax2.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
            # Customize the minor grid
            ax2.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
            if 'dice_cce_loss' in self.history.history:
                plt.plot(N, self.history.history["dice_cce_loss"], label="dice_cce_loss")
            if 'val_dice_cce_loss' in self.history.history:
                plt.plot(N, self.history.history["val_dice_cce_loss"], label="val_dice_cce_loss")
            plt.legend()

        path_filename = os.path.join(weight_dir, f"{self.experiment_name}_{self.net_name}_learning.png")
        plt.savefig(path_filename,
                    dpi=96, facecolor='w',
                    edgecolor='w', orientation='portrait',
                    format=None, transparent=False,
                    bbox_inches=None, pad_inches=0.1,
                    metadata=None
                    )
        plt.show()
        pass

    def figshow_matrix(self, ):
        classes = self.class_weights.keys()
        x_test, y_test = dataset.create_data_from_gen()
        # x_test, y_test = next(self.dataset.all_gen)
        y_pred = self.get_predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)
        matrix = confusion_matrix(y_test, y_pred, normalize='true')
        df_matrix = pd.DataFrame(matrix, range(3), range(3))
        plt.figure(figsize=(10,7))
        sns.set(font_scale=1.4)  # for label size
        sns.heatmap(df_matrix, annot=True, annot_kws={"size": 16})  # font size
        plt.show()
        # fig, ax = plt.subplots(figsize=(10, 10))
        # ax.set_title('Confusion Matrix')
        # disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=classes)
        # disp.plot(ax=ax)
        # plt.xticks(rotation=45)
        # plt.show()
        pass

if __name__ == "__main__":
    start = datetime.datetime.now()
    timezone = pytz.timezone("Europe/Moscow")
    image_size = 256
    batch_size = 40
    epochs = 20
    start_learning_rate = 0.001
    start_patience = round(epochs * 0.04)

    print(f'Image Size = {image_size}x{image_size}')
    dataset = ImagesDataSet(os.path.join(os.getcwd(), "data", "train", "images"),
                            os.path.join(os.getcwd(), "data", "train", "train.csv"),
                            image_size=image_size,
                            )
    dataset.batch_size = batch_size
    # dataset.augmentation = False
    dataset.validation_split = 0.15
    dataset.build()
    tr = TrainNN(dataset)
    tr.monitor = "val_loss"
    tr.learning_rate = start_learning_rate
    tr.es_patience = 20
    tr.rlrs_patience = start_patience
    tr.epochs = epochs
    tr.keras_model, tr.net_name = resnet50v2_classification_model(input_shape=(tr.dataset.image_size,
                                                                               tr.dataset.image_size) + (3,),
                                                                  num_classes=3)
    # tr.train()
    end = datetime.datetime.now()
    # print(f'Planned epochs: {epochs} Calculated epochs : {len(tr.history.history["loss"])} Time elapsed: {end - start}')
    # tr.figshow_base()

    """ Checking train on all available data """
    dataset.build_check_gen()
    tr.evaluate(dataset.all_gen)
    # dataset.build_check_gen()
    tr.figshow_matrix()
    print("ok")
