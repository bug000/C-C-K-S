# encoding: utf-8
from kashgari.utils.crf import CRF, crf_loss, crf_accuracy
from keras.engine import InputSpec
from keras.layers import Dense, Bidirectional, BatchNormalization, SpatialDropout1D, Dropout, CuDNNLSTM, interfaces
import keras.backend as K
from keras.models import Model

from kashgari.tasks.seq_labeling.base_model import SequenceLabelingModel


@interfaces.legacy_spatialdropout1d_support
class TimestepDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super(TimestepDropout, self).__init__(rate, **kwargs)
        self.input_spec = InputSpec(ndim=3)

    def _get_noise_shape(self, inputs):
        input_shape = K.shape(inputs)
        noise_shape = (input_shape[0], input_shape[1], 1)
        return noise_shape


class CRFModel(SequenceLabelingModel):
    __architect_name__ = 'CRFModel'
    __base_hyper_parameters__ = {
        'lstm_layer': {
            'units': 256,
            'return_sequences': True
        },
        'dense_layer': {
            'units': 64,
            'activation': 'tanh'
        }
    }

    def _prepare_model(self):
        base_model = self.embedding.model
        emb = base_model.output
        emb = SpatialDropout1D(0.3)(emb)
        crf_out = CRF(len(self.label2idx), sparse_target=False)(emb)
        self.model = Model(base_model.inputs, crf_out)

    def _compile_model(self):
        self.model.compile(loss=crf_loss,
                           optimizer='adam',
                           metrics=[crf_accuracy])


class CRFDropModel(SequenceLabelingModel):
    __architect_name__ = 'CRFDropModel'
    __base_hyper_parameters__ = {
        'lstm_layer': {
            'units': 256,
            'return_sequences': True
        },
        'dense_layer': {
            'units': 64,
            'activation': 'tanh'
        }
    }

    def _prepare_model(self):
        base_model = self.embedding.model
        emb = base_model.output
        emb = SpatialDropout1D(0.3)(emb)
        emb = TimestepDropout(0.1)(emb)
        blstm_out = Bidirectional(CuDNNLSTM(**self.hyper_parameters['lstm_layer']))(emb)
        blstm_out = BatchNormalization()(blstm_out)
        dense_ = Dense(512, activation='relu')(blstm_out)
        dense_ = SpatialDropout1D(0.5)(dense_)
        dense_ = TimestepDropout(0.2)(dense_)
        crf_out = CRF(len(self.label2idx), sparse_target=False)(dense_)
        self.model = Model(base_model.inputs, crf_out)

    def _compile_model(self):
        self.model.compile(loss=crf_loss,
                           optimizer='adam',
                           metrics=[crf_accuracy])

