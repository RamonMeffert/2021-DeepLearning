
import tensorflow as tf
import argparse
import numpy as np
from keras.datasets import cifar10
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,\
                                    Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model



# scale pixels
#taken from https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
def prep_pixels(train, test):
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

def create_res_net():

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


    return model

def main(args):
    if args.outdir:
        output_directory = args.outdir
    else:
        output_directory = './'
    if args.name:
        run_name = args.name
    elif args.optimizer:
        run_name = args.optimizer
    else:
        run_name = 'new_run'

    # load dataset
    (trainX, trainy), (testX, testy) = cifar10.load_data()

    trainX, testX = prep_pixels(trainX, testX)

    optimizers = ['sgd', 'adam', 'nadam']
    for optimizer in optimizers:
        model = create_res_net()
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        #model.summary()
        run_time = datetime.datetime.now().isoformat(timespec='minutes')
        path = os.path.join(log_dir, 'ResNet_train_' + optimizer + '_at_' + run_time + '.log')
        callbacks = [ CSVLogger(path, append=True, separator=',') ]

        model.fit(trainX, trainy,  epochs=20, batch_size=64, verbose=1)
        results = model.evaluate(testX, testy)[1]

        #save it
        run_time = datetime.datetime.now().isoformat(timespec='minutes')
        path = os.path.join(log_dir, 'ResNet_test_' + optimizer + '_at_' + run_time + '.log')
        eval_dict = dict(zip(self.model.metrics_names, evaluation))

        with open(path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            for key, val in eval_dict.items():
                writer.writerow([key, val])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', help='The output directory for log files.', type=str)
    parser.add_argument('-n', '--name', help='A semi-unique name for the run. Will be included in the generated log file to make it easier to find back results.', type=str)
    args = parser.parse_args()

    # ! When running on peregrine, output MUST go to /data/$USER/project/ or /home/$USER/project
    if args.outdir:
        print("Saving output to" + args.outdir)
        main(args)
    else:
        print("Saving output to default location")
        main("./")
