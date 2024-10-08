# Code for *Constructing Custom Thermodynamics Using Deep Learning*

***Note: a more modern implementation (in jax) for more general usage is now online at https://github.com/MLDS-NUS/onsagernet-jax. This current repository is purely for archival purposes.***

---

Code for reproduction of results in manuscript "Constructing Custom Thermodynamics Using Deep Learning".

## Datasets

Datasets are available in the Harvard Dataverse public repository
https://doi.org/10.7910/DVN/NRIX7Y

Description of the data contents
### Simulation Data
- `config_train_mean_every.pkl`: high-dimensional training data, consisting of 610 trajectories of polymer chains configurations (300 x 3 coordinates) over 1001 time steps (split into 5 files due to size, can be combined with software such as 7-zip)
- `config_test_mean_every.pkl`: high-dimensional test data, consisting of 110 trajectories of polymer chains configurations (300 x 3 coordinates) over 1001 time steps
- `DNA_data_resnet.pkl`: low-dimensional training and test data (3-dimensional)
- `ex_train.pkl`: chain extension (Z1) for training data
- `ex_test.pkl`: chain extension (Z1) for test data
- `config_fast_mean_test.pkl`: high-dimensional test data, consisting of 500 trajectories (same initial configuration) of polymer chains configurations (300 x 3 coordinates) over 1001 time steps with a fast unfolding time
- `config_middle_mean_test.pkl`: high-dimensional test data, consisting of 500 trajectories (same initial configuration) of polymer chains configurations (300 x 3 coordinates) over 1001 time steps with a middle unfolding time (split into 4 files due to size, can be combined with software such as 7-zip)
- `config_slow_mean_test.pkl`: high-dimensional test data, consisting of 500 trajectories (same initial configuration) of polymer chains configurations (300 x 3 coordinates) over 1001 time steps with a slow unfolding time

### Experimental Data
- `Dumbbell.tif`: raw experimental video images for molecule that takes on dumbbell configuration
- `Dumbbell.png`: processed experimental video images for molecule that takes on dumbbell configuration
- `Folded.tif`: raw experimental video images for molecule that takes on folded configuration
- `Folded.png`: processed experimental video images for molecule that takes on folded configuration
- `config_coord.pkl`: high-dimensional data for experimental images of molecules in stretched configurations from literature
- `X_low1.txt`: low-dimensional data for experimental images of molecules in stretched configurations from Soh et al. (2023)
- `X_low2.txt`: low-dimensional data for experimental images of molecules in stretched configurations from Soh et al. (2018)


## System Requirements

The code was tested on
- Ubuntu version 20.04
- Python version 3.8
- Tensorflow version 2.9.1 with tensorflow.compat.v1 as tf and tf.disable_v2_behavior()

## Installation Instructions

*(Typical install time: <5 minutes)*

Create a virtual environment
```bash
virtualenv -p python3 ~/.venvs/<env_name>
```

Activate the environment
```bash
source ~/.venvs/<env_name>/bin/activate
```

Install required python packages
```bash
pip install -r requirements.txt
```

## Code
Before running the codes, download the relevant - `Simulation` or `Experimental` datasets from https://doi.org/10.7910/DVN/NRIX7Y and save them into the `./data` directory.

### Quick Demo

*(Typical run time: 15 minutes for the default 5000 iterations)*

A quick demo of the code is provided in `./demo/`.
This demo is the copy of the code used to obtain the results presented in the paper, but the number of trajectories provided is limited to 250 and 50 for training and testing, respectively. This, along with the default 5000 iterations (this is user defined) ensures that the code can be run on most personal laptop or desktop computers in a short amount of time. The limited amount of data means that the prediction accuracy will differ from that obtained with full dataset, and the number of iterations chosen by user will also affect the quality of the predictions.

Run the demo by issuing
```bash
python ./demo/demo_OnsagerNet.py
```
The code will create an output folder, you can choose the name of this folder. If you do not input a name and press enter in the terminal, a default `./demo_outputs/` folder will be created.
Expected output in the demo output folder are the model checkpoints.

### Full Reproduction

The full reproduction code for the training of Stochastic OnsagerNet is provided `./reproduction/`.

#### Training

*(Typical run time: about 4 hours)*

Train the model by running the Jupyter notebook `/reproduction/OnsagerNet.ipynb`

The model checkpoints are saved to `./reproduction/checkpoints/`

#### Inference

*(Typical run time: about 10 minites)*

To produce main results presented in the paper
1. Trajectory prediction
    - prediction of fast trajectories: run the Jupyter notebook`./reproduction/prediction/predict_fast.ipynb`
    - prediction of middle trajectories: run the Jupyter notebook`./reproduction/prediction/predict_middle.ipynb`
    - prediction of slow trajectories: run the Jupyter notebook`./reproduction/prediction/predict_slow.ipynb`
2. Physical meaning of learned coordinates: run the Jupyter notebook`./reproduction/physical meaning/physical_meaning.ipynb`
3. learned potential energy landscape
    - projected onto Z1-Z2: run the Jupyter notebook`./reproduction/potential/potential_Z1Z2.ipynb`
    - projected onto Z1-Z3: run the Jupyter notebook`./reproduction/potential/potential_Z1Z3.ipynb`
    - projected onto Z2-Z3: run the Jupyter notebook`./reproduction/potential/potential_Z2Z3.ipynb`

For reproducibility, the default loads pre-trained checkpoints in `./reproduction/saved_checkpoints/`. Alternatvely, you may load your trained checkpoints in `./reproduction/checkpoints/`, which may result in slight statistical variations in the results.
