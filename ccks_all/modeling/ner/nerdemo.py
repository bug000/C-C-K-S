from kashgari.tasks.seq_labeling import BLSTMCRFModel
from kashgari.embeddings import BERTEmbedding
from keras.callbacks import EarlyStopping, TensorBoard

from ccks_ner import dload

train_x, train_y = dload.load_json_data('train')
validate_x, validate_y = dload.load_json_data('validate')
test_x, test_y = dload.load_json_data('test')

print(f"train data count: {len(train_x)}")
print(f"validate data count: {len(validate_x)}")
print(f"test data count: {len(test_x)}")

model_path = r"D:\data\biendata\ccks2019_el\ner\m0"
log_filepath = r"D:\data\biendata\ccks2019_el\ner\m0log"

# emn_path = r'D:\data\bert\chinese_L-12_H-768_A-12'
emn_path = r'D:\data\bert\chinese-bert_chinese_wwm_L-12_H-768_A-12'

# check_point = ModelCheckpoint(model_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min")

early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=2)
# early_stop = EarlyStopping(monitor="val_crf_accuracy", mode="max", patience=2)

log = TensorBoard(log_dir=log_filepath, write_images=False, write_graph=True, histogram_freq=0)

embedding = BERTEmbedding(emn_path, 50)

model = BLSTMCRFModel(embedding)

model.__base_hyper_parameters__ = {
        'lstm_layer': {
            'units': 256,
            'return_sequences': True
        },
        'dense_layer': {
            'units': 64,
            'activation': 'tanh'
        }
    }


model.fit(train_x,
          train_y,
          x_validate=validate_x,
          y_validate=validate_y,
          epochs=20,
          batch_size=512,
          labels_weight=True,
          fit_kwargs={'callbacks': [early_stop, log]})

model.evaluate(test_x, test_y)

model.save(model_path)
"""
ep20
                        precision    recall  f1-score   support
        fictionalhuman     0.7541    0.7700    0.7620       661
                tvshow     0.8809    0.9703    0.9234       404
                 place     0.8156    0.8581    0.8363      1402
                 thing     0.7811    0.7332    0.7564      5746
            vocabulary     0.8681    0.7880    0.8261     15190
                 movie     0.8214    0.9014    0.8596      1674
          organization     0.7618    0.7691    0.7654       524
                 human     0.8212    0.6310    0.7137      1805
   entertainmentperson     0.8067    0.9194    0.8594      2247
                tvplay     0.8573    0.9164    0.8859      1974
               country     0.9049    0.7804    0.8381       378
   communicationmedium     0.8486    0.7566    0.8000       941
                  tool     0.7596    0.7054    0.7315       112
          creativework     0.8124    0.8044    0.8084      3472
               athlete     0.6833    0.6212    0.6508        66
   collegeoruniversity     0.8429    0.8194    0.8310        72
              language     0.7654    0.8267    0.7949        75
                 event     0.8036    0.8422    0.8225       374
                  game     0.8550    0.8188    0.8365       425
      culturalheritage     0.9565    0.7857    0.8627        84
                  food     0.7778    0.5833    0.6667        48
               product     0.7348    0.6299    0.6783       154
      historicalperson     0.7159    0.7241    0.7200        87
              organism     0.0000    0.0000    0.0000        11
scientificorganization     0.8039    0.7321    0.7664        56
                 plant     0.7193    0.5616    0.6308        73
               dynasty     0.7255    0.7400    0.7327        50
                 brand     0.7917    0.4222    0.5507        45
    academicdiscipline     0.8125    0.7358    0.7723        53
                animal     0.7302    0.6571    0.6917        70
      medicalcondition     0.7500    0.4000    0.5217        15
            familyname     0.0000    0.0000    0.0000         5
        fictionalthing     1.0000    0.1765    0.3000        17
              currency     0.0000    0.0000    0.0000         9
      awardeventseries     0.9091    0.6667    0.7692        15
                nation     0.7692    0.5556    0.6452        18
              material     0.7500    0.2812    0.4091        32
    astronomicalobject     0.0000    0.0000    0.0000        10
        educationmajor     0.5714    0.5333    0.5517        15
                person     0.7500    0.2857    0.4138        21
            curriculum     0.0000    0.0000    0.0000         3
      historicalperiod     0.0000    0.0000    0.0000         9
                symbol     0.0000    0.0000    0.0000         3
            zodiacsign     0.0000    0.0000    0.0000         2
            realestate     0.0000    0.0000    0.0000         7
 medicaldepartmenttype     0.0000    0.0000    0.0000         1
       chemicalelement     0.0000    0.0000    0.0000         1
               theorem     0.0000    0.0000    0.0000         1
             micro avg     0.8317    0.7917    0.8112     38457
             macro avg     0.8313    0.7917    0.8091     38457

"""