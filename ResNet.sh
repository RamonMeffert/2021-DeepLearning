#!/bin/bash

#SBATCH --job-name="alexnet-relu-adam"

# Send an email when important events happen
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=r.a.meffert@student.rug.nl

# Run for at most 1 hour
#SBATCH --time=01:00:00

# Run on 1 gpu, doesn't matter what type (no preference for v100 or k40)
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Clean environment
module purge

# Load everything we need for TensorFlow (loads python, tensorflow and a lot more) and scikit
module load TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4 scikit-learn/0.22.2.post1-fosscuda-2019b-Python-3.7.4

# Train the network
python ~/DL/p1/2021-DeepLearning/ResNet.py --outdir ~/DL/p1/2021-DeepLearning/ \
                                           --name 'ResNet'
