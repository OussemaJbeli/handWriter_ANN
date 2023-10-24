import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()


X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

X_train = np.expand_dims(X_train, axis=-1)  
X_test = np.expand_dims(X_test, axis=-1)  


datagen = ImageDataGenerator(
    rotation_range=10, 
    width_shift_range=0.1, 
    height_shift_range=0.1, 
    zoom_range=0.1
)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))


optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy']
            )


early_stopping = EarlyStopping(monitor='val_loss', patience=3)


model.fit(datagen.flow(X_train, y_train, batch_size=32), 
            epochs=20, validation_data=(X_test, y_test), 
            callbacks=[early_stopping])


val_loss, val_acc = model.evaluate(X_test, y_test)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_acc)

model.save('training.model')
