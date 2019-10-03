<!--<h3><b>Cat Mouse Game with TensorFlow DQN Plugin</b></h3>-->
## <b>Cat Mouse Game with TensorFlow DQN Plugin</b> [[GitHub Repository]](https://github.com/tdesfont/TensorFlow-DQN-CatMouseGame) <br>

![Teaser Image](https://wildlifetv.files.wordpress.com/2014/09/tom-jerry-chase.jpg)

## Exploration-exploitation in DQN:

PyTorch and TensorFlow implementation of a RL experiment. Our toy example is a prey-predator scheme where both are bounded in a restricted squared area. The predator is going straight to the prey (Greedy strategy). The prey is given a set of four actions available (Up/Down/Left/Right). Without learning any policy, the prey is doing a simple random walk.

The goal of our experiments is to learn with Deep-Q Network an optimal strategy for the prey (the learning agent) to flee from the predator in the constrained area.

A reasonable assumption to make is that if the agent adopts for strategy, a maximization of the distance prey/predator, it will be stuck at a corner in the long-run. It is therefore necessary for the agent to by explorative. Intuitivelly, there is an optimal strategy for the prey which is to do a periodic movement near the borders indefinitelly. We shall see if this policy can be learned or not.

We will focus on Deep-Q Network as learning algorithms. In a first try, they will be used as function approximation for the Q-Function.

## Prerequisites
All the scripts should be started from Anaconda and Spyder3 due
to Tkinter usage.
______________________________________________________________________

## WalkThrough
______________________________________________________________________

Once biocells/ and display/ are set up, you can execute the following test scripts:
    test_verbose_biocells.py
    test_deterministic_policy_biocells.py
    test_display_biocells.py

Test: test_verbose_biocells.py
Generates one game simulation with random action for our prey.
Print the distance between prey and predator and game events.

Test: test_deterministic_policy_biocells.py
Generates one game simulation with the specified policy: Up and Left chosen at random with proba 1/2. 
Display the interactive simulation.
Need display/ to be functional.

Test: test_display_biocells.py
Display one game simulation with random actions for the agent.
______________________________________________________________________

Now we add the matplotlib_vision/ feature that try to  roughly imitate the way OpenAI works with input image of Atari games. The idea is to convert a matplotlib image to an RGB array. This will allow easy manipulation of input game frames for the DQN or any computer vision based network. Also note that there is more information by taking the difference between a screen and its previous one. Simply because like in numerical computations we have an estimation of the speed and the directions of the agent and predator.

Test: test_differential_screen_vision.py
PreRequisite: Due to issues with Tkinter, this script needs to be executed from Spyder3/Anaconda. It is likely not to work started from a shell.
Generate the screens and differential screen as RGB arrays (images). A sample for one game simulation is in test_matplotlib_vision/. You can compare for the same game the original somewhat static screen or differential screens.
______________________________________________________________________

Once our environment and our inputs are all set, it is time to add a model to solve the learning task. The first try is in PyTorch and is a direct application of the PyTorch Tutorial on DQN but with our own input. 

Test: test_train_torch_model.py
Train the torch model defined whose DQN net structure location is torch_model/. You can check TensorBoard by starting a new shell and going to the following address on your web-browser to see the loss evolution or the lifetime that we'd like to maximise for example: 
http://localhost:6006/

Test: test_torch_model_evaluate.py
Do one run of the game with the trained network. No yet functional.
______________________________________________________________________

To handle the continuous state-action space (due to continuity of state), we implement discretization of the state space by a converting scalar positions of prey and predator to a grid. By using histograms, we can choose a arbitrary number of bins to discretize the state space. Another intersting idea implemented in the Pytorch tutorial was to keep information of the previous states. To do so, the network input will consist of the weighted sum of a certain number of temporal grids.

Test: test_discretization.py
Run a game and save discretized screens in /test_discretization.
The grid_1 folder shows the grid as a matrix of shape (h, w, 1) meaning that the presence of the prey and predator is indicated by +1 or -1. This is not handy due to possible positions overlap in the histogram.
We will use an image matrix of shape (h, w, 3) with the convention for information encodage in the pixel: [1, 0, 0] means the prey is in this bin. [0, 1, 0] means the predator is in the bin and [1, 1, 0] means that the prey and predator are in the same bin and the game has likely ended.
______________________________________________________________________
    
TensorFlow model (training and evaluation of the model)

Test: test_tensorflow_policy_training.py
______________________________________________________________________

In that new folder we create a new prioritized replay with simpler tree search than MCTS to force the network to learn on the critical situation. 

______________________________________________________________________


######################################################################

## Repository structure

RL-Exploration-Exploitation/

    biocells/               # The RL learning context and experiment (Step, Reward, Play, Move)
    display/                # Interactive matplotlib visualisation
    matplotlib_vision/      # Generation of screen (for computer vision as Atari/OpenAI do)
    runs/                   # TensorBoardX runs
    tensorflow_model/       # Model attempt in TensorFlow
    torch_model/            # Model attempt in PyTorch
    
    test_matplotlib_vision/  
    
    test_deterministic_policy_biocells.py
    test_differential_screen_vision.py
    test_display_biocells.py
    test_matplotlib_vision
    test_torch_model_evaluate.py
    test_train_torch_model.py


    
