import tensorflow as tf
from tensorflow.keras import backend
import tensorflow.keras.metrics

__version__ = 0.12


class DiceCoefficient(tensorflow.keras.metrics.Metric):

    def __init__(self, name='dice_coef', **kwargs):
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.dice: float = 0
        self.short_name = name
        pass

    def update_state(self, y_true, y_pred, smooth=1, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # по размеру тензора определяем пришли маски по изображениям или по тексту
        shape = y_true.shape
        shape_len = len(shape)
        if shape_len == 2:
            axis = [1]
        elif shape_len == 3:
            axis = [1, 2]
        elif shape_len == 4:
            axis = [1, 2, 3]
        intersection = backend.sum(y_true * y_pred, axis=axis)
        union = backend.sum(y_true, axis=axis) + backend.sum(y_pred, axis=axis)
        self.dice = backend.mean((2. * intersection + smooth) / (union + smooth), axis=0)

    def result(self):
        return self.dice

    def __str__(self):
        return self.short_name

    def reset_state(self):
        self.dice: float = 0
        pass


class DiceCCELoss(tensorflow.keras.metrics.Metric):
    def __init__(self, name='dice_cce_loss', **kwargs):
        super(DiceCCELoss, self).__init__(name=name, **kwargs)
        self.dice: float = 0
        self.cce_loss: float = 0
        self.short_name = name
        self.cce = tf.keras.losses.CategoricalCrossentropy()
        pass

    def update_state(self, y_true, y_pred, smooth=1, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # по размеру тензора определяем пришли маски по изображениям или по тексту
        shape = y_true.shape
        shape_len = len(shape)
        if shape_len == 2:
            axis = [1]
        elif shape_len == 3:
            axis = [1, 2]
        elif shape_len == 4:
            axis = [1, 2, 3]
        self.cce_loss = self.cce(y_true, y_pred)
        intersection = backend.sum(y_true * y_pred, axis=axis)
        union = backend.sum(y_true, axis=axis) + backend.sum(y_pred, axis=axis)
        self.dice = 1.0 - (backend.mean((2. * intersection + smooth) / (union + smooth), axis=0))
        self.dice = self.cce_loss + self.dice

    def result(self):
        return self.dice

    def __str__(self):
        return self.short_name

    def reset_state(self):
        self.dice: float = 0
        pass


class CustomF1Score(tensorflow.keras.metrics.Metric):
    def __init__(self, name='f1_score_custom', **kwargs):
        super(CustomF1Score, self).__init__(name=name, **kwargs)
        self.weighted_f1_score: float = 0.0
        self.short_name = name

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Count positive samples.
        TP = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
        pred_positives = backend.sum(backend.round(backend.clip(y_pred, 0, 1)))
        positives = backend.sum(backend.round(backend.clip(y_true, 0, 1)))
        # How many selected items are relevant?
        precision = TP / (pred_positives + backend.epsilon())
        # How many relevant items are selected?
        recall = TP / (positives + backend.epsilon())
        # Calculate f1_score
        f1_score = 2 * (precision * recall) / (precision + recall + backend.epsilon())

        self.weighted_f1_score = f1_score * TP / backend.sum(TP)
        self.weighted_f1_score = backend.sum(self.weighted_f1_score)

    def result(self):
        return self.weighted_f1_score

    def __str__(self):
        return self.short_name

    def reset_state(self):
        self.weighted_f1_score: float = 0.
