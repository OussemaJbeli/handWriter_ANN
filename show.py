import matplotlib.pyplot as plt

import tensorflow as tf

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Choose an index to display a specific image

for index in range(10):
# Display the image
    plt.imshow(x_train[index], cmap='gray')
    plt.title('Label: {}'.format(y_train[index]))
    plt.show()
