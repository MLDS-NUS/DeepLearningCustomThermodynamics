# Code for *Constructing Custom Thermodynamics Using Deep Learning*

Code for reviewers of manuscript "Constructing Custom Thermodynamics Using Deep Learning".

*Note: This repository contains code to reproduce the construction of thermodynamic coordinates and training the stochastic OnsagerNet as presented in the paper. Note that this project was started during Tensorflow 1.XX's maintainence timeframe but we are in the process of migrating it into an updated implementation in Tensorflow 2.XX. The current implementation is for review purposes only, and we will release the more user-friendly Tensorflow 2 version (currently work in progress) for easy adaptation to general problems upon publication.*

## Datasets

Datasets are available in the Harvard Dataverse public repository
https://doi.org/10.7910/DVN/NRIX7Y

Description of the data contents
### Simulation Data
- `config_train_mean_every.pkl`: high- dimensional training data, consisting of 610 trajectories of polymer chains configurations (300 x 3 coordinates) over 1001 time steps (split into 5 files due to size, can be combined with software such as 7-zip)
- `config_test_mean_every.pkl`: high-dimensional test data, consisting of 110 trajectories of polymer chains configurations (300 x 3 coordinates) over 1001 time steps
- `DNA_data_resnet.pkl`: low-dimensional training and test data (3-dimensional)
- `ex_train.pkl`: chain extension (Z1) for training data
- `ex_test.pkl`: chain extension (Z1) for test data

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
- Ubuntu version **XYZ**
- Python version 3.**XYZ**
- Tensorflow version 1.**XYZ**
- CUDA version **XYZ**

## Installation Instructions

*(Typical install time: **XYZ**)*

Create a virtual environment
```bash
virtualenv ~/.venvs/<env_name> -p 3.**XYZ**
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

### Quick Demo

*(Typical run time: **XYZ**)*

A quick demo of the code is provided in `./demo/`.
This demo (**Describe the demo and how it relates to full code**).

Run the demo by issuing
```bash
python ./demo/**XYZ**.py
```

Expected output: **XYZ**

### Full Reproduction

The full reproduction code for the training of Stochastic OnsagerNet is provided `./reproduction/`.

#### Training

*(Typical run time: **XYZ**)*

Download data files `**XYZ**` from https://doi.org/10.7910/DVN/NRIX7Y and save them into the `./data` directory.

Train the model by running the Jupyter notebook `/reproduction/**XYZ**.ipynb`

The model checkpoints are saved to `./reproduction/checkpoints/`

#### Inference

*(Typical run time: **XYZ**)*

To produce main results presented in the paper, run the Jupyter notebook `./reproduction/**XYZ**.ipynb`.

For reproducibility, the default loads pre-trained checkpoints in `./reproduction/saved_checkpoints/`. Alternatvely, you may load your trained checkpoints in `./reproduction/checkpoints/`, which may result in slight statistical variations in the results.
