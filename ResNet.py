import tensorflow as tf
import numpy as np

import datetime
import argparse
import os
import csv

from tensorflow import Tensor
from tensorflow.python.keras.callbacks import History, CSVLogger
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10


class ResNet:
    def __init__(self, activation_function = 'relu', optimizer = 'adam'):
        """Create a new instance of an ResNet model.

        Args:
            activation_function (str, optional): The activation function. Must be one of the activation functions available in Keras. Defaults to 'relu'.
            optimizer (str, optional): The optimizer. Must be one of the optimizers available in Keras. Defaults to 'adam'.
        """
        # Set the seed for NumPy
        np.random.seed(1000)

        # Set the activation function
        self.activation_function = activation_function

        # Set the optimizer
        self.optimizer = optimizer

        # Init the model
        self.model = self.__create_model()


    def __create_model(self) -> Sequential:
        """Initialize the ResNet model with the parameters given. (private function)

        Returns:
            Sequential: A compiled model
        """
        inputs = Input(shape=(32, 32, 3))
        num_filters = 64

        t = BatchNormalization()(inputs)
        t = Conv2D(kernel_size=3,
                strides=1,
                filters=num_filters,
                padding="same")(t)
        t = relu_bn(t)

        num_blocks_list = [2, 5, 5, 2]
        for i in range(len(num_blocks_list)):
            num_blocks = num_blocks_list[i]
            for j in range(num_blocks):
                t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
            num_filters *= 2

        t = AveragePooling2D(4)(t)
        t = Flatten()(t)
        outputs = Dense(10, activation='softmax')(t)

        model = Model(inputs, outputs)

        print("Set up model, compiling...")

        model.compile(loss = tf.keras.losses.sparse_categorical_crossentropy, optimizer= self.optimizer, metrics=['accuracy'])

        print("Compilation done")

        return model


    def train(self, training_data, training_labels, validation_data, validation_labels, run_name, batch_size = 64, epochs = 10, log = False, log_dir = './logs/') -> History:
        """Train the model.

        Args:
            data (NumPy array): The images to train on
            labels (NumPy array): The labels of the images
            batch_size (int, optional): The batch size. Defaults to 32.
            epochs (int, optional): The number of epochs to train for. Defaults to 10.
            log (bool, optional): Whether to log the results of training

        Returns:
            History: Information about the model training. Useful for analysis and making graphs.
        """

        # If we want to log the results, use Keras' callback functionality to write to a CSV file
        # This gives us loss and accuracy data for the training and the validation data
        if log:
            run_time = datetime.datetime.now().isoformat(timespec='minutes')
            path = os.path.join(log_dir, 'resnet_train_' + run_name + '_at_' + run_time + '.log')
            callbacks = [ CSVLogger(path, append=True, separator=',') ]
        else:
            callbacks = []

        return self.model.fit(
            x               = training_data,
            y               = training_labels,
            validation_data = (validation_data, validation_labels),
            batch_size      = batch_size,
            epochs          = epochs,
            callbacks       = callbacks
        )


    def test(self, data, labels, run_name, log = False, log_dir = './logs/'):
        """Evaluate the model. Only makes sense after the model is trained.

        Args:
            data (Numpy array): Images
            labels (Numpy array): Labels

        Returns:
            scalar | [scalar]: The results from evaluating the model
        """

        print("Evaluating the model")

        # Use Keras' built in evaluation. This has a callback parameter as well,
        # but for some reason that doesn't seem to work
        evaluation = self.model.evaluate(
            x         = data, 
            y         = labels
        )

        # Write output to file if needed
        if log:
            run_time = datetime.datetime.now().isoformat(timespec='minutes')
            path = os.path.join(log_dir, 'resnet_test_' + run_name + '_at_' + run_time + '.log')
            eval_dict = dict(zip(self.model.metrics_names, evaluation))

            with open(path, 'w') as csvfile:
                writer = csv.writer(csvfile)
                for key, val in eval_dict.items():
                    writer.writerow([key, val])


def prep_pixels(train, test):
    """Scales pixels so instead of being integers in [0, 255] they are floats in [0, 1].
    Based on taken from https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/

    Args:
        train ([type]): [description]
        test ([type]): [description]

    Returns:
        [type]: [description]
    """
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm


def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn


def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out


def main(args):
    if args.outdir:
        output_directory = args.outdir
    else:
        output_directory = './'
    
    if args.epochs:
        epochs = args.epochs
    else:
        epochs = 50
    
    if args.optimizer:
        optimizer = args.optimizer
    else:
        optimizer = 'adam'
    
    if args.activation:
        activation_function = args.activation
    else:
        activation_function = 'relu'

    if args.name:
        run_name = args.name
    elif args.optimizer:
        run_name = args.optimizer
    else:
        run_name = 'new_run'

    # Load CIFAR10 Data set
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    #Train-validation-test split
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.3)

    # Create ResNet instance
    resnet = ResNet(
        activation_function = activation_function,
        optimizer           = optimizer
    )

    # Train ResNet
    resnet.train(
        training_data     = x_train,
        training_labels   = y_train,
        validation_data   = x_val,
        validation_labels = y_val,
        epochs            = epochs,
        log               = True,
        log_dir           = output_directory,
        run_name          = run_name
    )

    # Evaluate ResNet
    resnet.test(
        data     = x_test, 
        labels   = y_test,
        log      = True,
        log_dir  = output_directory,
        run_name = run_name
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', help='The output directory for log files.', type=str)
    parser.add_argument('-e', '--epochs', help='The number of epochs to train the model for.', type=int)
    parser.add_argument('-m', '--optimizer', help='The optimizer to use.', type=str)
    parser.add_argument('-a', '--activation', help='The activation function to use.', type=str)
    parser.add_argument('-n', '--name', help='A semi-unique name for the run. Will be included in the generated log file to make it easier to find back results.', type=str)
    args = parser.parse_args()

    # ! When running on peregrine, output MUST go to /data/$USER/project/ or /home/$USER/project
    main(args)
