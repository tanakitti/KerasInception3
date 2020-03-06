# python3 train.py --train train --validation validation --epoch 1

import pandas as pd
import numpy as np
import os

import argparse

import keras
from keras.models import Sequential
from keras.layers import Dense,GlobalAveragePooling2D, Dropout
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.models import model_from_yaml

ap = argparse.ArgumentParser()
ap.add_argument('-t', '--train', required=True,
	help='path to the train image floder')
ap.add_argument('-v', '--validation', required=True,
	help='path to the validation image floder')
ap.add_argument('-e', '--epoch', type=int, default=50,
	help='number of epoches')

args = vars(ap.parse_args())


base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output

## add some layer ##
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

train_datagen = ImageDataGenerator(
  preprocessing_function=preprocess_input,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	brightness_range=(1, 1.3),
	horizontal_flip=True,
	vertical_flip=True,
	fill_mode='nearest')

train_generator=train_datagen.flow_from_directory(args['train'],
                                                 target_size=(299,299),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
step_size_train=train_generator.n//train_generator.batch_size
model.summary()

classFile = open('Classes.txt','w') 
for c in train_generator.class_indices.keys():
  classFile.write('c\n') 
classFile.close()                              

model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=args['epoch'])


test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = test_datagen.flow_from_directory(args['validation'],
                                                        target_size=(299,299),
                                                        color_mode='rgb',
                                                        batch_size=32,
                                                        class_mode='categorical',
                                                        shuffle=False)

# file validation picture
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(args['validation']):
    for file in f:
        if '.jpg' in file:
            files.append(os.path.join(r, file))

Y_pred = model.predict_generator(validation_generator, len(files) // 32+1)

y_pred = np.argmax(Y_pred, axis=1)
y_test = validation_generator.classes

print(y_pred)
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))
print('Classification Report')
target_names = train_generator.class_indices.keys()
print(classification_report(y_test, y_pred, target_names=target_names))
modelAcc = accuracy_score(y_test, y_pred)

# save all model model
model.save('Inception3.'+str(modelAcc)+'.h5')

# save model weight
model.save_weights('Inception3.'+str(modelAcc)+'.h5')

# save model layers
model_yaml = model.to_yaml()
with open('Inception3.'+str(modelAcc)+'.yaml', 'w') as yaml_file:
    yaml_file.write(model_yaml)