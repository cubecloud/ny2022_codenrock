import os
import time
import pytz
import datetime
import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import LearningRateScheduler

__version__ = 0.001


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
                 start_learning_rate=1e-4,
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
        self.start_learning_rate = start_learning_rate

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
            self.best = np.Inf
        else:
            msg = f'Error: mode is unknown! -> {self.mode}'
            assert self.mode == 'max' or self.mode == 'min', msg

        if self.monitor_mode == 'validation':
            self.monitor = self.val_metric_name
        elif self.monitor_mode == 'train':
            self.monitor = self.metric
        msg = f'Error: monitor_mode is unknown! -> {self.mode}'
        assert self.monitor_mode == 'validation' or self.monitor_mode == 'train', msg

    def checkLR(self, epoch, lr):
        if (epoch <= int(self.patience / 2)) or (epoch < 7):
            return lr
        diff = 0.08
        decay1 = 0.955
        decay2 = 1.03
        point1_max = ((self.history['mae'][-3] + self.history['mae'][-1]) / 2) * (1 + diff)
        point1_min = ((self.history['mae'][-3] + self.history['mae'][-1]) / 2) * (1 - diff)
        point2_max = ((self.history['mae'][-4] + self.history['mae'][-2]) / 2) * (1 + diff)
        point2_min = ((self.history['mae'][-2] + self.history['mae'][-2]) / 2) * (1 - diff)

        print(point1_max, point1_min, point2_max, point2_min)
        if (point2_max <= point1_max and point2_min <= point1_min) or (
                point2_max >= point1_max and point2_min >= point1_min):
            print(f'Decay1: {lr:0.7f} * {decay1} = {lr * decay1:0.7f}')
            lr = lr * decay1
        elif (np.mean(self.history['mae'][-6:]) > self.history['mae'][-1] * diff) and (
                np.mean(self.history['mae'][-6:]) < self.history['mae'][-1] / diff):
            print(f'Decay2: {lr:0.7f} * {decay2} = {lr * decay2:0.7f}')
            lr = lr * decay2
        return lr

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        pass

    def on_epoch_start(self, epoch, lr, logs=None):
        print(f'\r{epoch} mae {logs["mean_absolute_error"]:7.2f}', end='')

    def __patience_extender(self, epoch):
        if (epoch >= 10) and (epoch % 10 == 0):
            self.patience += round((self.epochs - epoch) * 0.03)
            # print (f'Patience: {self.patience}')

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = datetime.datetime.now(self.timezone)
        self.epoch_start = time.time()

    # def on_epoch_end(self, epoch, logs=None):
        # pred = self.keras_model.predict(
        #     [xTrainScaled[valMask], xTrainC01[valMask]])  # Полуаем выход сети на проверочно выборке
        # predUnscaled = yScaler.inverse_func(
        #     pred).flatten()  # Делаем обратное нормирование выхода к изначальным величинам цен квартир
        # yTrainUnscaled = yScaler.inverse_func(
        #     yTrainScaled[valMask]).flatten()  # Делаем такое же обратное нормирование yTrain к базовым ценам

        # delta = predUnscaled - yTrainUnscaled  # Считаем разность предсказания и правильных цен
        # absDelta = abs(delta)  # Берём модуль отклонения
        # print(f'MaxOut index {np.argmax(absDelta)}')

        # current_mae = logs.get('mae')
        # current_val_mae = logs.get('val_mae')

        # print(
        #     f'\rEpoch: {epoch} - absolute error: {round(sum(absDelta) / (1e+6 * len(absDelta)), 3)} - mae: {current_mae:5.4f} - val_mae: {current_val_mae:5.4f}',
        #     end='')

    def on_epoch_end(self, epoch, logs=None):
        current_metric = logs.get(self.monitor)
        if self.mode_function(self.monitor, self.best):
            self.best = current_metric
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print('Restoring model weights from the end of the best epoch.')
                self.model.set_weights(self.best_weights)
                filename = f'{self.save_prefix}_{self.monitor}.h5'
                self.model.save_weights(os.path.join(self.save_dir, filename))
                print(f'Epoch: {epoch} - {self.monitor} {logs[self.monitor]:7.2f}')

        self.epoch_finished = time.time()
        self.epoch_end_time = datetime.datetime.now(self.timezone)
        msg = f'- {round(self.epoch_finished - self.epoch_start, 2)}sec \n' \
              f'Time elapsed: {self.epoch_end_time - self.training_start} - ' \
              f'ETA: {(((self.epoch_end_time - self.training_start) / (epoch + 1)) * (self.epochs - (epoch + 1)))}'
        print(msg)

        for key, val in logs.items():
            if key in self.history:
                self.history[key].append(val)
            else:
                self.history[key] = [val]
        # self.figshow(self.history)
        # time.sleep(5.5)
        pass

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f'\nEpoch {self.stopped_epoch + 1}: early stopping')
            self.finished_time = time.time()
        pass

    # def figshow(self, history):
    #     fig = plt.figure(figsize=(18, 6))
    #     sns.set_style("white")
    #     ax1 = fig.add_subplot(1, 3, 1)
    #     ax1.set_axisbelow(True)
    #     ax1.minorticks_on()
    #     ax1.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
    #     ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    #     N = np.arange(0, len(history["loss"]))
    #     plt.ylim([-0.01, 0.3])
    #     plt.plot(N, history["loss"], linestyle='-', color='red', label="Training loss")
    #     plt.plot(N, history["val_loss"], linestyle='--', color='blue', label="Validation loss")
    #     plt.plot(N, history["val_mae"], linestyle='-', color='cyan', label="val_mean_absolute_error")
    #     if 'lr' in history:
    #         lr_list = [x * 100 for x in history["lr"]]
    #         plt.plot(N, lr_list, linestyle=':', color='green', label="lr * 1000")
    #     plt.title(f"Training/validation Loss and Mean Squared Error")
    #     plt.legend()
    #     ax2 = fig.add_subplot(1, 3, 2)
    #     ax2.set_axisbelow(True)
    #     ax2.minorticks_on()
    #     ax2.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
    #     ax2.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    #     pred = model.predict([xTrainScaled[valMask], xTrainC01[valMask]])
    #     predUnscaled = yScaler.inverse_func(pred).flatten()
    #     yTrainUnscaled = yScaler.inverse_func(yTrainScaled[valMask]).flatten()
    #     ax2.scatter(yTrainUnscaled, predUnscaled, c='blue')
    #     plt.xlabel('True Values price')
    #     plt.ylabel('Pred Values price')
    #     plt.axis('equal')
    #     plt.axis('square')
    #     plt.xlim([0, plt.xlim()[1]])
    #     plt.ylim([0, plt.ylim()[1]])
    #     ax2.plot([0, 1], [0, 1], transform=ax2.transAxes, color='red')
    #     ax3 = fig.add_subplot(1, 3, 3)
    #     ax3.set_axisbelow(True)
    #     ax3.minorticks_on()
    #     ax3.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
    #     ax3.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    #     ax3.hist(history["val_loss"][-3:], density=True, bins=3, color='orange', lw=1, histtype='stepfilled', alpha=0.8,
    #              label='val_loss')
    #     ax3.hist(history["val_mae"][-3:], density=True, bins=3, color='red', lw=1, histtype='stepfilled', alpha=0.8,
    #              label='val_mae')
    #     ax3.hist(history["loss"][-3:], density=True, bins=3, color='skyblue', lw=1, histtype='stepfilled', alpha=0.6,
    #              label='loss')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()
    #     pass

    # def on_train_batch_begin(self, batch, logs=None):
    #   print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now(timezone).time()))

if __name__ == '__main__':
    electro = ElectroF1()
    lrs = LearningRateScheduler(electro.checkLR, verbose=0)
