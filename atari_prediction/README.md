# Atari Prediction

This problem is based on reinforcement learning. Given the observation made by an agent, we are attempting to predict the return that it will obtain. The agents are playing Atari games with a prespecified array of actions. We precompute the returns to prevent having to iterate through all of the actions twice.

Much of the code is based on sample_test.py provided by Esraa Elelimy.

## Datasets

We are using actions from Khurram Javed's Atari prediction benchmark. They are provided for various Atari games supported by the Gym library.

## Base Models
 - Value Network - 3 convolutional layers and 2 dense layers
 - Large Model - 4 blocks of 2-3 convolutional layers followed by pooling. Then 3 dense layers with optional dropout in between. Batch normalization after each convolutional or dense layer.