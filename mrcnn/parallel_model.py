"""
Mask R-CNN
Multi-GPU Support for Keras.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

Ideas and a small code snippets from these sources:
https://github.com/fchollet/keras/issues/2436
https://medium.com/@kuza55/transparent-multi-gpu-training-on-tensorflow-with-keras-8b0016fd9012
https://github.com/avolkov1/keras_experiments/blob/master/keras_exp/multigpu/
https://github.com/fchollet/keras/blob/master/keras/utils/training_utils.py
"""

import tensorflow as tf
import keras.backend as K
import keras.layers as KL
import keras.models as KM


class ParallelModel(KM.Model):
    """Subclasses the standard Keras Model and adds multi-GPU support.
    It works by creating a copy of the model on each GPU. Then it slices
    the inputs and sends a slice to each copy of the model, and then
    merges the outputs together and applies the loss on the combined
    outputs.
    """

    def __init__(self, keras_model, gpu_count):
        """Class constructor.
        keras_model: The Keras model to parallelize
        gpu_count: Number of GPUs. Must be > 1
        """
        merged_outputs = self.make_parallel(keras_model, gpu_count)
        super(ParallelModel, self).__init__(inputs=keras_model.inputs,
                                            outputs=merged_outputs)
        self.inner_model = keras_model
        self.gpu_count = gpu_count

    def __getattribute__(self, attrname):
        """Redirect loading and saving methods to the inner model. That's where
        the weights are stored."""
        if 'load' in attrname or 'save' in attrname:
            return getattr(self.inner_model, attrname)
        return super(ParallelModel, self).__getattribute__(attrname)

    def summary(self, *args, **kwargs):
        """Override summary() to display summaries of both, the wrapper
        and inner models."""
        super(ParallelModel, self).summary(*args, **kwargs)
        self.inner_model.summary(*args, **kwargs)

    def make_parallel(self, inner_model,gpu_count):
        """Creates a new wrapper model that consists of multiple replicas of
        the original model placed on different GPUs.
        """
        def get_slice(data, i, parts):
            shape = tf.shape(data)
            batch_size = shape[:1]
            input_shape = shape[1:]
            step = batch_size // parts
            if i == gpu_count - 1:
                size = batch_size - step * i
            else:
                size = step
            size = tf.concat([size, input_shape], axis=0)
            stride = tf.concat([step, input_shape * 0], axis=0)
            start = stride * i
            return tf.slice(data, start, size)

        output_names = inner_model.output_names
        outputs_all = []
        for i in range(len(inner_model.outputs)):
            outputs_all.append([])

        # Run the model call() on each GPU to place the ops there
        for i in range(gpu_count):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('tower_%d' % i):
                    inputs = []
                    # Retrieve a slice of the input.
                    for x in inner_model.inputs:
                        # In-place input splitting which is not only
                        # 5% ~ 12% faster but also less GPU memory
                        # duplication.
                        with tf.device(x.device):
                            input_shape = K.int_shape(x)[1:]
                            slice_i = KL.Lambda(get_slice,
                                                output_shape=input_shape,
                                                arguments={'i': i,
                                                           'parts': gpu_count})(x)
                            inputs.append(slice_i)
                    # Create the model replica and get the outputs
                    outputs = inner_model(inputs)
                    if not isinstance(outputs, list):
                        outputs = [outputs]
                    # Save the outputs for merging back together later
                    for l, o in enumerate(outputs):
                        outputs_all[l].append(o)

        # Merge outputs on CPU
        with tf.device('/cpu:0'):
            merged = []
            for outputs, name in zip(outputs_all, output_names):
                # Concatenate or average outputs?
                # Outputs usually have a batch dimension and we concatenate
                # across it. If they don't, then the output is likely a loss
                # or a metric value that gets averaged across the batch.
                # Keras expects losses and metrics to be scalars.
                if K.int_shape(outputs[0]) == ():
                    # Average
                    m = KL.Lambda(lambda o: tf.add_n(o) / len(outputs), name=name)(outputs)
                else:
                    # Concatenate
                    m = KL.Concatenate(axis=0, name=name)(outputs)
                merged.append(m)
        return merged


if __name__ == "__main__":
    # Testing code below. It creates a simple model to train on MNIST and
    # tries to run it on 2 GPUs. It saves the graph so it can be viewed
    # in TensorBoard. Run it as:
    #
    # python3 parallel_model.py

    import os
    import numpy as np
    import keras.optimizers
    from keras.datasets import mnist
    from keras.preprocessing.image import ImageDataGenerator

    GPU_COUNT = 2

    # Root directory of the project
    ROOT_DIR = os.path.abspath("../")

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    def build_model(x_train, num_classes):
        # Reset default graph. Keras leaves old ops in the graph,
        # which are ignored for execution but clutter graph
        # visualization in TensorBoard.
        tf.reset_default_graph()

        inputs = KL.Input(shape=x_train.shape[1:], name="input_image")
        x = KL.Conv2D(32, (3, 3), activation='relu', padding="same",
                      name="conv1")(inputs)
        x = KL.Conv2D(64, (3, 3), activation='relu', padding="same",
                      name="conv2")(x)
        x = KL.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
        x = KL.Flatten(name="flat1")(x)
        x = KL.Dense(128, activation='relu', name="dense1")(x)
        x = KL.Dense(num_classes, activation='softmax', name="dense2")(x)

        return KM.Model(inputs, x, "digit_classifier_model")

    # Load MNIST Data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype('float32') / 255
    x_test = np.expand_dims(x_test, -1).astype('float32') / 255

    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    # Build data generator and model
    datagen = ImageDataGenerator()
    model = build_model(x_train, 10)

    # Add multi-GPU support.
    model = ParallelModel(model, GPU_COUNT)

    optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=5.0)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])

    model.summary()

    # Train
    model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=64),
        steps_per_epoch=50, epochs=10, verbose=1,
        validation_data=(x_test, y_test),
        callbacks=[keras.callbacks.TensorBoard(log_dir=MODEL_DIR,
                                               write_graph=True)]
    )