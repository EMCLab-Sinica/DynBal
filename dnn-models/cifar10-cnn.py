import pathlib

import onnx
from tensorflow.keras import datasets, losses, layers, models
import tf2onnx.convert

from model_utils import remove_tensorflow_input_transpose

def main():
    # Simplified from https://www.tensorflow.org/tutorials/images/cnn

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
                  loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10,
              validation_data=(test_images, test_labels))

    models_dir = pathlib.Path(__file__).resolve().parent
    onnx_model, _ = tf2onnx.convert.from_keras(model, opset=11)
    onnx_model = remove_tensorflow_input_transpose(onnx_model)
    onnx.save(onnx_model, models_dir / 'cifar10-cnn.onnx')

if __name__ == '__main__':
    main()
