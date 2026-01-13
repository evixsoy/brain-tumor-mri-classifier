import tensorflow as tf

#picks first gpu for training
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True) 
tf.keras.mixed_precision.set_global_policy('mixed_float16')

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input

#preparing images for resnet50
train_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rotation_range= 20,
    width_shift_range=0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip= True,
    fill_mode = 'nearest'
)

#only for test,valid dataset
test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

import os 
output_folder = os.path.join('dataset_split')

train_dir = os.path.join(output_folder, 'train')
val_dir = os.path.join(output_folder, 'val')
test_dir = os.path.join(output_folder, 'test')

image_size = (256,256)
batch_size = 32

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size = image_size,
    batch_size = batch_size,
    class_mode = 'categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size = image_size,
    batch_size = batch_size,
    class_mode = 'categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size = image_size,
    batch_size = batch_size,
    class_mode = 'categorical'
)

from keras.applications.resnet import ResNet50

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3)) # 3 = rgb

base_model.trainable= False

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from keras import layers, models

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation = 'relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax', dtype='float32')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=12,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        'best_model.keras',
        save_best_only=True,
        monitor='val_loss'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5
    ),
    CSVLogger('training_history.csv')
]

model.fit(train_data, epochs =75, validation_data=val_data, callbacks=callbacks)
model.save("base_model.keras")


