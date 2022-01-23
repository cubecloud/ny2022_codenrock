import os
import time
import pytz
import datetime
import numpy as np
from tensorflow.keras.callbacks import Callback

__version__ = 0.002


class ElectroF1(Callback):
    def __init__(self,
                 logs=None,
                 epochs: int = 0,
                 save_dir: str = '',
                 save_prefix: str = '',
                 metric_name: str = 'f1_score',
                 mode: str = 'max',
                 monitor_mode: str = 'validation',
                 keras_model=None,
                 patience=15,
                 save_best_epoch: bool = True,
                 # start_learning_rate=1e-4,
                 ):

        super(Callback, self).__init__()
        self.version = "clbk_v01"
        self.timezone = pytz.timezone("Europe/Moscow")
        self.save_prefix = save_prefix
        self.epochs = epochs
        self.save_dir = save_dir
        self.metric_name = metric_name
        self.val_metric_name = f'val_{self.metric_name}'
        self.mode = mode
        self.monitor_mode = monitor_mode
        self.keras_model = keras_model
        self.patience = patience
        self.save_best_epoch = save_best_epoch
        # self.start_learning_rate = start_learning_rate

        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self.history = dict()
        self.training_start = datetime.datetime.now(self.timezone)
        self.start_time = None
        self.epoch_start = None
        self.stopped_epoch = None
        self.epoch_finished = None
        self.epoch_end_time = None
        self.finished_time = None
        self.wait = None

        if self.mode == 'max':
            self.mode_function = np.greater
            # Initialize the best as negative infinity.
            self.best = np.NINF

        elif self.mode == 'min':
            self.mode_function = np.less
            # Initialize the best as positive infinity.
            self.best = np.PINF
        else:
            msg = f'Error: mode is unknown! -> {self.mode}'
            assert self.mode == 'max' or self.mode == 'min', msg

        self.best_epoch: int = 0

        if self.monitor_mode == 'validation':
            self.monitor = self.val_metric_name
        elif self.monitor_mode == 'train':
            self.monitor = self.metric
        msg = f'Error: monitor_mode is unknown! -> {self.mode}'
        assert self.monitor_mode == 'validation' or self.monitor_mode == 'train', msg

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        pass

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = datetime.datetime.now(self.timezone)
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        current_metric = logs.get(self.monitor)
        if isinstance(current_metric, np.ndarray):
            current_metric = np.mean(current_metric)

        if self.mode_function(current_metric, self.best):
            self.best = current_metric
            self.wait = 0
            # Record the best weights if current results is better.
            msg = f'Memorizing best model weights from the end of the epoch: {epoch} - {current_metric:7.4f}'
            print(msg)
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch
            if self.save_best_epoch:
                msg = f'Saving weights'
                print(msg)
                filename = f'{self.save_prefix}_{self.metric_name}.h5'
                self.model.save_weights(os.path.join(self.save_dir, filename))
                print(f'Epoch: {epoch} - {self.monitor} {current_metric:7.4f}')
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print('Restoring model weights from the end of the best epoch.')
                self.model.set_weights(self.best_weights)
                print(f'Best epoch: {self.best_epoch} - {self.monitor} {self.best:7.4f}')

        self.epoch_finished = time.time()
        self.epoch_end_time = datetime.datetime.now(self.timezone)
        # f'{round(self.epoch_finished - self.epoch_start, 2)}sec '
        eta_time = (((self.epoch_end_time - self.training_start) / (epoch + 1)) * (
                    self.epochs - (epoch + 1))) + datetime.datetime.now(self.timezone)
        msg = f'Time elapsed: {self.epoch_end_time - self.training_start} - ' \
              f'ETA: {eta_time}'
        print(msg)

        for key, val in logs.items():
            if key in self.history:
                self.history[key].append(val)
            else:
                self.history[key] = [val]
        pass

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f'\nEpoch {self.stopped_epoch + 1}: early stopping')
            self.finished_time = time.time()
        pass
