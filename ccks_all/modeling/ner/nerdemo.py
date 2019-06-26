from kashgari.tasks.seq_labeling import BLSTMCRFModel
from kashgari.embeddings import BERTEmbedding
from keras.callbacks import EarlyStopping, TensorBoard

from ccks_ner import dload

train_x, train_y = dload.load_json_data_no_type('train')
validate_x, validate_y = dload.load_json_data_no_type('validate')
test_x, test_y = dload.load_json_data_no_type('test')

print(f"train data count: {len(train_x)}")
print(f"validate data count: {len(validate_x)}")
print(f"test data count: {len(test_x)}")

model_path = r"D:\data\biendata\ccks2019_el\ner\m0not"
log_filepath = r"D:\data\biendata\ccks2019_el\ner\m0notlog"

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
          epochs=100,
          batch_size=1024,
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



ep40
                        precision    recall  f1-score   support

                 movie     0.9283    0.9510    0.9395      1674
   entertainmentperson     0.9319    0.9568    0.9442      2247
                 thing     0.8970    0.8656    0.8811      5746
          creativework     0.8807    0.9248    0.9022      3472
   communicationmedium     0.9177    0.9245    0.9211       941
        fictionalhuman     0.9396    0.8941    0.9163       661
            vocabulary     0.8961    0.9065    0.9012     15190
                symbol     0.0000    0.0000    0.0000         3
                  game     0.9476    0.9365    0.9420       425
                tvplay     0.9009    0.9762    0.9370      1974
                  tool     0.9043    0.9286    0.9163       112
                 human     0.8973    0.8809    0.8890      1805
                tvshow     0.9450    0.9777    0.9611       404
                 place     0.9577    0.9215    0.9393      1402
               product     0.9203    0.8247    0.8699       154
          organization     0.8996    0.9065    0.9030       524
               country     0.8821    0.9497    0.9146       378
                 event     0.9160    0.9626    0.9387       374
                 brand     0.9524    0.8889    0.9195        45
      culturalheritage     1.0000    0.9643    0.9818        84
              language     0.9583    0.9200    0.9388        75
                person     0.9524    0.9524    0.9524        21
      awardeventseries     1.0000    1.0000    1.0000        15
                 plant     0.9559    0.8904    0.9220        73
                animal     0.9844    0.9000    0.9403        70
               athlete     0.8551    0.8939    0.8741        66
                  food     0.9057    1.0000    0.9505        48
              organism     1.0000    0.8182    0.9000        11
      historicalperson     0.8208    1.0000    0.9016        87
    academicdiscipline     0.8983    1.0000    0.9464        53
   collegeoruniversity     0.9697    0.8889    0.9275        72
        fictionalthing     0.9167    0.6471    0.7586        17
      medicalcondition     0.9286    0.8667    0.8966        15
scientificorganization     0.8710    0.9643    0.9153        56
        educationmajor     0.8750    0.9333    0.9032        15
               dynasty     0.8909    0.9800    0.9333        50
              material     0.8571    0.9375    0.8955        32
                nation     0.8889    0.8889    0.8889        18
              currency     0.8571    0.6667    0.7500         9
      historicalperiod     0.8333    0.5556    0.6667         9
            realestate     1.0000    0.7143    0.8333         7
            familyname     0.5000    0.2000    0.2857         5
    astronomicalobject     1.0000    0.8000    0.8889        10
            zodiacsign     1.0000    1.0000    1.0000         2
       chemicalelement     1.0000    1.0000    1.0000         1
 medicaldepartmenttype     0.0000    0.0000    0.0000         1
            curriculum     1.0000    0.6667    0.8000         3
               theorem     0.0000    0.0000    0.0000         1

             micro avg     0.9038    0.9120    0.9079     38457
             macro avg     0.9040    0.9120    0.9077     38457




















"""

