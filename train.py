import os
from typing import Tuple
import pytz
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import seaborn as sns
from models import resnet50v2_original_model
from dataset import ImagesDataSet
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

__version__ = 0.007

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
        self.experiment_name = f"{self.dataset.version}_tr_{__version__}"
        self.history = None
        self.epochs = 15

        self.batch_size = None
        self.monitor = "categorical_accuracy"
        self.loss = "categorical_crossentropy"
        # self.metrics = [CustomF1Score(), tfa.metrics.F1Score(num_classes=dataset.num_classes)]
        self.metrics = [tfa.metrics.F1Score(num_classes=dataset.num_classes)]
        self.path_filename: str = ''
        self.model_compiled = False
        self.es_patience = 15
        self.rlrs_patience = 8
        self.keras_model, self.net_name = resnet50v2_original_model(input_shape=(self.dataset.image_size,
                                                                                 self.dataset.image_size) + (3,),
                                                                    num_classes=3)

        self.learning_rate = 3e-5
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.class_weights = self.dataset.class_weights

    def compile(self):
        self.path_filename = os.path.join(weight_dir, f"{self.experiment_name}_{self.net_name}_{self.monitor}")
        self.keras_model.summary()
        self.keras_model.compile(optimizer=self.optimizer,
                                 loss=self.loss,
                                 metrics=self.metrics,
                                 )
        self.model_compiled = True
        pass

    def train(self):
        if not self.model_compiled:
            self.compile()
        chkp = ModelCheckpoint(f"{self.path_filename}.h5",
                               mode='auto',
                               monitor=self.monitor,
                               save_best_only=True,
                               )
        rlrs = ReduceLROnPlateau(monitor=self.monitor, factor=0.8, patience=self.rlrs_patience, min_lr=1e-07)
        es = EarlyStopping(patience=self.es_patience, monitor=self.monitor, restore_best_weights=True, verbose=1)
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

    def fine_train(self):
        self.compile()
        self.load_best_weights()
        chkp = ModelCheckpoint(f"{self.path_filename}"+"_at_{epoch:02d}.h5",
                               save_freq='epoch',
                               save_weights_only=True,
                               verbose=1
                               )
        callbacks = [chkp]

        self.history = self.keras_model.fit(self.dataset.all_gen,
                                            epochs=self.epochs,
                                            verbose=1,
                                            callbacks=callbacks,
                                            class_weight=self.class_weights
                                            )
        self.model_compiled = True
        pass

    def load_best_weights(self, path_filename=None):
        if not path_filename:
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

    def figshow_base(self, save_figure=True, show_figure=True):
        def num_of_zeros(n):
            s = '{:.16f}'.format(n).split('.')[1]
            return len(s) - len(s.lstrip('0'))
        sub_plots = 1
        plot_names_list = list(self.history.history.keys())
        subplot2_list = ['mae', 'mse', 'f1', 'dice_cce_loss']
        to_plot2 = set()
        for plot_name in plot_names_list:
            for subplot2_name in subplot2_list:
                if subplot2_name in plot_name:
                    to_plot2.add(plot_name)
        to_plot1 = set(plot_names_list).difference(to_plot2)

        if to_plot2:
            sub_plots = 2
        fig = plt.figure(figsize=(30, 9*sub_plots))
        sns.set_style("white")
        ax1 = fig.add_subplot(1, sub_plots, 1)
        ax1.set_axisbelow(True)
        ax1.minorticks_on()
        ax1.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        N = np.arange(0, len(self.history.history["loss"]))

        for plot_name in to_plot1:
            if "lr" in plot_name:
                mult = 10 ** (num_of_zeros(self.learning_rate)+2)
                lr_arr = np.array(self.history.history[plot_name])
                lr_arr = lr_arr * mult//10 if mult > 2 else lr_arr * mult
                plt.plot(N, lr_arr, label=f"{plot_name}*{mult}")
            else:
                plt.plot(N, self.history.history[plot_name], label=plot_name)
        plt.title(f"Training Loss and Metric")
        plt.legend()
        if sub_plots == 2:
            ax2 = fig.add_subplot(1, sub_plots, sub_plots)
            ax2.set_axisbelow(True)
            ax2.minorticks_on()
            ax2.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
            ax2.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
            for plot_name in to_plot2:
                if "f1" in plot_name:
                    f1_arr = np.array(self.history.history[plot_name])
                    if len(f1_arr.shape) > 1:
                        f1_arr = f1_arr.mean(axis=1)
                    plt.plot(N, f1_arr, label=plot_name)
                else:
                    plt.plot(N, self.history.history[plot_name], label=plot_name)
            plt.legend()

        if save_figure:
            path_filename = os.path.join(weight_dir, f"{self.experiment_name}_{self.net_name}_learning.png")
            plt.savefig(path_filename,
                        dpi=96, facecolor='w',
                        edgecolor='w', orientation='portrait',
                        format=None, transparent=False,
                        bbox_inches=None, pad_inches=0.1,
                        metadata=None
                        )
        if show_figure:
            plt.show()
        pass

    def figshow_matrix(self, ):
        classes = list(self.class_weights.keys())
        dataset.build_check_gen(batch_size=1280)
        x_test, y_test = next(self.dataset.all_gen)
        y_pred = self.get_predict(x_test)
        y_pred = np.expand_dims(np.argmax(y_pred, axis=1), axis=1)
        y_test = np.expand_dims(np.argmax(y_test, axis=1), axis=1)
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title('Confusion Matrix')
        cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        cm_disp.plot(ax=ax)
        plt.xticks(rotation=45)
        plt.show()
        pass

if __name__ == "__main__":
    start = datetime.datetime.now()
    timezone = pytz.timezone("Europe/Moscow")
    image_size = 448
    batch_size = 24
    epochs = 140
    start_learning_rate = 0.00005
    start_patience = round(epochs * 0.04)

    print(f'Image Size = {image_size}x{image_size}')
    dataset = ImagesDataSet(train_dir,
                            os.path.join(base_dir, "train.csv"),
                            image_size=image_size,
                            )
    dataset.batch_size = batch_size
    dataset.validation_split = 0.1
    dataset.build()
    tr = TrainNN(dataset)
    tr.monitor = "loss"
    tr.learning_rate = start_learning_rate
    tr.es_patience = 22
    tr.rlrs_patience = start_patience
    tr.epochs = epochs

    tr.train()
    end = datetime.datetime.now()
    print(f'Planned epochs: {epochs} Calculated epochs : {len(tr.history.history["loss"])} Time elapsed: {end - start}')
    tr.figshow_base(save_figure=True, show_figure=True)

    """ Checking train on all available data """
    dataset.build_check_gen(batch_size=batch_size)
    tr.evaluate(dataset.all_gen)
    """ Check confusion matrix """
    tr.figshow_matrix()

    dataset.build_check_gen(batch_size=batch_size, shuffle=True, augmentation=True)
    tr.learning_rate = 1e-7
    tr.epochs = 12
    tr.compile()
    # tr.load_best_weights(path_filename='/home/cubecloud/Python/projects/ny2022_codenrock/data/weight/ds_v5_tr_0.007_ResNet50V2_imagenet_3_448x448_loss_at_06.h5')
    tr.fine_train()

    dataset.build_check_gen(batch_size=batch_size)
    tr.evaluate(dataset.all_gen)
    tr.figshow_matrix()

    print("ok")
