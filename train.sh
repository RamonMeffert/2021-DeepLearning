#!/bin/bash

#SBATCH --job-name="example"

# Send an email when important events happen
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=r.a.meffert@student.rug.nl

# Run for at most 1 hour
#SBATCH --time=01:00:00

# Run on 1 gpu, doesn't matter what type (no preference for v100 or k40)
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Copy training data to temp storage for faster file access
cp /data/$USER/data-dl-meffert-gassilloud.tar.gz /local/tmp/

# Change directory to local tmp
cd /local/tmp

# Extract data
tar xvzf data-dl-meffert-gassilloud.tar.gz

# Go back to working directory
cd ~

# Clean environment
module purge

# Load everything we need for TensorFlow (loads python, tensorflow and a lot more)
module load TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4

# Train the network
python ~/deep_learning_course/project_1/train_network.py

# Clean up
rm -r /local/tmp/data-dl-meffert-gassilloud
rm /local/tmp/data-dl-meffert-gassilloud.tar.gz
