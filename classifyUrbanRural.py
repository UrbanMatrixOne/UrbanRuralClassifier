from keras.applications.mobilenetv2 import preprocess_input
from keras.preprocessing import image
import keras.applications 
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import optimizers


import pandas as pd
import numpy as np
import os, sys, re 

from absl import flags


#WEIGHT_PATH = './weights/best_weights.hdf5'
LABELS = {0: 'Biomass',
 1: 'Coal',
 2: 'Gas',
 3: 'Geothermal',
 4: 'Hydro',
 5: 'Nuclear',
 6: 'Oil',
 7: 'Other',
 8: 'Solar',
 9: 'Waste',
 10: 'Wind'}

flags.DEFINE_string('input_dir', None, 'Directory that contains the images to classify')
flags.DEFINE_string('output_file', None, 'csv to create with output')
flags.DEFINE_string('model_weight_path', './weights/best_weights.hdf5', 'path to pretrained model weights')
FLAGS = flags.FLAGS
FLAGS(sys.argv)

def main():
    print('hi')
    flags.mark_flag_as_required('input_dir')
    flags.mark_flag_as_required('output_file')

    base_model = keras.applications.mobilenetv2.MobileNetV2(weights='imagenet', include_top=False)
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # add a fully-connected layer
    x = Dense(1280, activation='relu')(x)
    # and a logistic layer for 11 classes
    predictions_layer = Dense(11, activation='softmax')(x)
    #assemble model
    model = Model(inputs=base_model.input, outputs=predictions_layer)

    #load pre-trained weights

    model.load_weights(FLAGS.model_weight_path)
    model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.001, momentum=0.9), metrics=["accuracy"])

    #data generator:
    predict_datagen = image.ImageDataGenerator(preprocessing_function = preprocess_input)
    predict_generator =predict_datagen.flow_from_directory(FLAGS.input_dir, class_mode = None,shuffle = False, batch_size = 1)
    predict_generator.reset()

    #make predictions
    predictions = model.predict_generator(predict_generator)

    #process results
    predicted_class_indices=np.argmax(predictions,axis=1)
    predicted_class_prob=np.max(predictions,axis=1)
    labelled_predictions = [LABELS[k] for k in predicted_class_indices]
    image_metadata = pd.DataFrame([s for s in [re.findall(r'-?\d+\.?\d*',filename) for filename in predict_generator.filenames]], 
             columns=('lon','lat','zoom'))
    results = pd.DataFrame({'Filename':predict_generator.filenames,
                'Prediction': labelled_predictions, 
                'probability': predicted_class_prob,
                'lon': image_metadata.lon, 
                'lat': image_metadata.lat,
                'zoom': image_metadata.zoom})

    #save to csv
    results.to_csv(FLAGS.output_file,index = False )

if __name__ == '__main__':
  main()




