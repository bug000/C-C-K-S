from kashgari.layers import LSTMLayer
from kashgari.tasks.classification import ClassificationModel
from keras import Model
from keras.layers import Bidirectional, Dense


class BLSTMModel(ClassificationModel):
    __architect_name__ = 'BLSTMModel'
    __base_hyper_parameters__ = {
        'lstm_layer': {
            'units': 256,
            'return_sequences': False
        },
        'activation_layer': {
            'activation': 'softmax'
        },
        'optimizer': {
            'module': 'keras.optimizers',
            'name': 'Adam',
            'params': {
                'lr': 1e-3,
                'decay': 0.0
            }
        },
        'compile_params': {
            'loss': 'categorical_crossentropy',
            # 'optimizer': 'adam',
            'metrics': ['accuracy']
        }
    }

    def _prepare_model(self):
        base_model = self.embedding.model
        blstm_layer = Bidirectional(LSTMLayer(**self.hyper_parameters['lstm_layer']))(base_model.output)
        dense_layer = Dense(len(self.label2idx), **self.hyper_parameters['activation_layer'])(blstm_layer)
        output_layers = [dense_layer]

        self.model = Model(base_model.inputs, output_layers)

    def _compile_model(self):
        optimizer = getattr(eval(self.hyper_parameters['optimizer']['module']),
                            self.hyper_parameters['optimizer']['name'])(
            **self.hyper_parameters['optimizer']['params'])
        self.model.compile(optimizer=optimizer, **self.hyper_parameters['compile_params'])





