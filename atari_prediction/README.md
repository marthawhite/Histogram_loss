# Atari Prediction

This problem is based on reinforcement learning. Given the observation made by an agent, we are attempting to predict the return that it will obtain. The agents are playing Atari games with a prespecified array of actions. We precompute the returns to prevent having to iterate through all of the actions twice.

Much of the code is based on sample_test.py provided by Esraa Elelimy.

## Datasets

We are using actions from the [Atari Prediction Benchmark](https://github.com/khurramjaved96/atari-prediction-benchmark) ([Javed *et al.* 2023](https://khurramjaved.com/scalable_rnns.pdf)). They are provided for various Atari games supported by the [Gym](https://www.gymlibrary.dev/index.html) library. Note: The new version has 100+ million actions per game, but we used an older version with 10-13 million.

## Base Models
 - Value Network - 3 convolutional layers and 2 dense layers
 - Large Model - 4 blocks of 2-3 convolutional layers followed by pooling. Then 3 dense layers with optional dropout in between. Batch normalization after each convolutional or dense layer.

 ## Instructions
 1. Set up your Python 3.10 environment using `requirements.txt`
 2. Precompute the returns for each game you are interested in running by running

    ```
    python precompute.py actions_dir actions_file returns_dir
    ```
    where `actions_dir` is the path to the directory containing the actions files, `actions_file` is the name of the actions file, and `returns_dir` is the path to the directory to save the returns in. The actions file should be named `{game}.txt` where game is the name of the Gym environment (e.g. `PongNoFrameskip-v4`), and the output file will be named `{game}.npy`.
3. Copy `main.py` to the project (outer) directory.
4. Train and evaluate the model by running
    ```
    python main.py actions_path returns_path
    ```
    where `actions_path` and `returns_path` are the paths to the actions and returns files respectively, and the actions file should be named as per the instructions in step 2.
5. Collect your training progression results in `hlg.json` and `reg.json` for HL-Gaussian and $\ell_2$ respectively. The test metrics will be saved together in `results.json`.

Feel free to refer to the provided Slurm batch scripts as examples of how to precompute returns and train the models.