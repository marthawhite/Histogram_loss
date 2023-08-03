# Atari Prediction

This problem is based on reinforcement learning. Given the observation made by an agent, we are attempting to predict the return that it will obtain. The agents are playing Atari games with a prespecified array of actions. We precompute the returns to prevent having to iterate through all of the actions twice.

Much of the code is based on sample_test.py provided by Esraa Elelimy.

## Datasets

We are using actions from the [Atari Prediction Benchmark](https://github.com/khurramjaved96/atari-prediction-benchmark) ([Javed *et al.* 2023](https://khurramjaved.com/scalable_rnns.pdf)). They are provided for various Atari games supported by the [Gym](https://www.gymlibrary.dev/index.html) library. Note: The new version has 100+ million actions per game, but we used an older version with 10-13 million.

## Base Models
 - Value Network - 3 convolutional layers and 2 dense layers
 - Large Model - 4 blocks of 2-3 convolutional layers followed by pooling. Then 3 dense layers with optional dropout in between. Batch normalization after each convolutional or dense layer.