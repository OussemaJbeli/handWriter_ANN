import os
import matplotlib.pyplot as plt
import tensorflow as tf

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Create the directory if it does not exist
os.makedirs('dataset_imgs/', exist_ok=True)

# Save the first 10 images in the specified directory without displaying them
for index in range(1000):
    # Create a new figure
    plt.figure()

    # Display the image
    plt.imshow(x_train[index], cmap='gray')
    plt.title('Label: {}'.format(y_train[index]))

    # Save the image with a unique filename in the specified directory
    plt.savefig('dataset_imgs/image_{}.png'.format(index))

    # Close the figure to free up memory
    plt.close()
