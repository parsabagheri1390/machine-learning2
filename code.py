import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from keras import layers
from keras.layers import GlobalAveragePooling2D,Dense,Dropout
from keras.applications import MobileNetV2
from keras import Input,Model
import tensorflow as tf

train_ds = '/mnt/c/Users/ASUS/Downloads/archive (11)/seg_train/seg_train'
test_ds = '/mnt/c/Users/ASUS/Downloads/archive (11)/seg_test/seg_test'
yhat = '/mnt/c/Users/ASUS/Downloads/archive (11)/seg_pred/seg_pred'

base_model=MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights=None
)

base_model.load_weights("/mnt/c/Users/ASUS/Downloads/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5")
base_model.trainable=False

inputs=keras.Input(shape=(224,224,3))
x=base_model(inputs,training=False)
x=GlobalAveragePooling2D()(x)
x=Dense(128,activation='relu')(x)
x=Dense(128,activation='relu')(x)
x=Dropout(0.3)(x)
output=Dense(6,activation='softmax')(x)
model=keras.Model(inputs=inputs,outputs=output)
        
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'])

test_data=tf.keras.utils.image_dataset_from_directory(
    test_ds,
    image_size=(224,224),
    label_mode='categorical',
    shuffle=False,
    batch_size=64
)

train_data=tf.keras.utils.image_dataset_from_directory(
    train_ds,
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=(224,224),
    label_mode='categorical',
    shuffle=True,
    batch_size=64
)
val_data=tf.keras.utils.image_dataset_from_directory(
    train_ds,
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=(224,224),
    label_mode='categorical',
    shuffle=False,
    batch_size=64
)
def preprocess(image,label):
    image=tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image,label

train_data=train_data.map(preprocess)
test_data=test_data.map(preprocess)
val_data=val_data.map(preprocess)

train_data=train_data.prefetch(1)
test_data=test_data.prefetch(1)
val_data=val_data.prefetch(1)
model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    batch_size=32)

set_train=False
model.trainable=True
for layer in base_model.layers:
    if layer.name=='block_13_expand':
        subset=True
    layers.trainable=subset
    

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    metrics=['accuracy']
    )

model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    batch_size=32,
)

model.evaluate(test_data)
