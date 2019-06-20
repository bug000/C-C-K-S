from keras.callbacks import Callback
import numpy as np
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score

from keras import backend as K


class Metrics(Callback):
    def __init__(self, y_index: int = 2):
        super().__init__()
        self.y_index = y_index
        self.val_recalls = []
        self.val_f1s = []
        self.val_precisions = []

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        pred = self.model.predict(self.validation_data[:self.y_index], verbose=1)
        val_predict = np.round(np.argmax(pred, axis=1)).astype(int)

        val_targ = self.validation_data[self.y_index]
        val_targ = np.round(np.argmax(val_targ, axis=1)).astype(int)

        _val_f1 = f1_score(val_targ, val_predict, average='micro')
        _val_recall = recall_score(val_targ, val_predict, average=None)  ###
        _val_precision = precision_score(val_targ, val_predict, average=None)  ###
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)

        report = classification_report(val_targ, val_predict)

        print(report)

        # print("— val_f1: %f — val_precision: %f — val_recall: %f" %(_val_f1, _val_precision, _val_recall))
        print("— val_f1: %f " % _val_f1)
        return


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


