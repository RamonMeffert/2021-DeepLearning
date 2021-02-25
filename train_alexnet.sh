#!/bin/bash

#SBATCH --job-name="alexnet-relu-adam"

# Send an email when important events happen
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=r.a.meffert@student.rug.nl

# Run for at most 1 hour
#SBATCH --time=01:00:00

# Run on 1 gpu, doesn't matter what type (no preference for v100 or k40)
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1

# Clean environment
module purge

# Load everything we need for TensorFlow (loads python, tensorflow and a lot more) and scikit
module load TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4 scikit-learn/0.22.2.post1-fosscuda-2019b-Python-3.7.4

# Train the network
python ~/deep_learning_course/project_1/AlexNet.py --outdir ~/deep_learning_course/project_1/ \
                                                   --epochs 50 \
                                                   --optimizer 'adam' \
                                                   --activation 'relu' \
                                                   --name 'adam-relu' \
                                                   --dropout 'reg'
