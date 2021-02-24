import tensorflow as tf
import numpy as np

import datetime
import argparse
import os
import csv

from tensorflow.python.keras.callbacks import History, CSVLogger
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, GaussianDropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class AlexNet:
    def __init__(self, activation_function = 'relu', optimizer = 'adam'):
        """Create a new instance of an AlexNet model.

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

        # Print the model summary
        # self.model.summary()

    def __create_model(self) -> Sequential:
        """Initialize the AlexNet model with the parameters given. (private function)

        Returns:
            Sequential: A compiled model
        """
        model = Sequential([            
            # 1st Convolutional Layer
            Conv2D(filters=96, input_shape=(32,32,3), kernel_size=(11,11), strides=(4,4), padding='same'),
            BatchNormalization(),
            Activation(self.activation_function),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),

            # 2nd Convolutional Layer
            Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'),
            BatchNormalization(),
            Activation(self.activation_function),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
        
            # 3rd Convolutional Layer
            Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'),
            BatchNormalization(),
            Activation(self.activation_function),

            # 4th Convolutional Layer
            Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'),
            BatchNormalization(),
            Activation(self.activation_function),

            #5th Convolutional Layer
            Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'),
            BatchNormalization(),
            Activation(self.activation_function),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),

            #Passing it to a Fully Connected layer
            Flatten(),
            # 1st Fully Connected Layer
            Dense(4096, input_shape=(32,32,3,)),
            BatchNormalization(),
            Activation(self.activation_function),
            # Add Dropout to prevent overfitting
            GaussianDropout(0.4),

            #2nd Fully Connected Layer
            Dense(4096),
            BatchNormalization(),
            Activation(self.activation_function),
            #Add Dropout
            GaussianDropout(0.4),

            #3rd Fully Connected Layer
            Dense(1000),
            BatchNormalization(),
            Activation(self.activation_function),
            #Add Dropout
            GaussianDropout(0.4),

            #Output Layer
            Dense(10),
            BatchNormalization(),
            Activation('softmax')
        ])

        print("Set up model, compiling...")

        model.compile(loss = tf.keras.losses.categorical_crossentropy, optimizer= self.optimizer, metrics=['accuracy'])

        print("Compilation done")

        return model

    def train(self, training_data, training_labels, validation_data, validation_labels, run_name, batch_size = 32, epochs = 10, log = False, log_dir = './logs/') -> History:
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
            path = os.path.join(log_dir, 'alexnet_train_' + run_name + '_at_' + run_time + '.log')
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
            path = os.path.join(log_dir, 'alexnet_test_' + run_name + '_at_' + run_time + '.log')
            eval_dict = dict(zip(self.model.metrics_names, evaluation))

            with open(path, 'w') as csvfile:
                writer = csv.writer(csvfile)
                for key, val in eval_dict.items():
                    writer.writerow([key, val])


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
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    #Train-validation-test split
    x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=.3)

    # One hot encoding
    y_train = to_categorical(y_train)
    y_val   = to_categorical(y_val)
    y_test  = to_categorical(y_test)

    # Create AlexNet instance
    alexnet = AlexNet(
        activation_function = activation_function,
        optimizer           = optimizer
    )

    # Train AlexNet
    alexnet.train(
        training_data     = x_train,
        training_labels   = y_train,
        validation_data   = x_val,
        validation_labels = y_val,
        epochs            = epochs,
        log               = True,
        log_dir           = output_directory,
        run_name          = run_name
    )

    # Evaluate AlexNet
    alexnet.test(
        data   = x_test, 
        labels = y_test,
        log    = True,
        log_dir = output_directory,
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
    if args.outdir:
        print("Saving output to" + args.outdir)
        main(args)
    else:
        print("Saving output to default location")
        main("./")