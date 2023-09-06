# Code for *Constructing Custom Thermodynamics Using Deep Learning*

Code for reviewers of manuscript "Constructing Custom Thermodynamics Using Deep Learning".

*Note: This repository contains code to reproduce the construction of thermodynamic coordinates and training the stochastic OnsagerNet as presented in the paper. Note that this project was started during Tensorflow 1.XX's maintainence timeframe but we are in the process of migrating it into an updated implementation in Tensorflow 2.XX. The current implementation is for review purposes only, and we will release the more user-friendly Tensorflow 2 version (currently work in progress) for easy adaptation to general problems upon publication.*

## Datasets

Datasets are available in the Harvard Dataverse public repository
https://doi.org/10.7910/DVN/NRIX7Y

Description of the data contents
- `config_train_mean_every.pkl`: high dimensional training data, consisting of **XYZ** trajectories of polymer chains configurations (300 x 3 coordinates) over **XYZ** time steps
- `config_test_mean_every.pkl`: high-dimensional test data, consisting of **XYZ** trajectories of polymer chains configurations (300 x 3 coordinates) over **XYZ** time steps
- `ex_train800.pkl`: **Add description**
- `ex_test800.pkl`: **Add description**

## System Requirements

The code was tested on
- Ubuntu version **XYZ**
- Python version 3.**XYZ**
- Tensorflow version 1.**XYZ**
- CUDA version **XYZ**

## Installation Instructions

*(Typical install time: **XYZ**)*

Install `Tensorflow` and optionally GPU support by following the instructions [here](https://www.tensorflow.org/install).

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

*(Typical install time: **XYZ**)*

A quick demo of the code is provided in `./demo/`.
This demo (**Describe the demo and how it relates to full code**).

Run the demo by issuing
```bash
python ./demo/**XYZ**.py
```

Expected output: **XYZ**

Expected run-time: **XYZ**

### Full Reproduction

The full reproduction code for the training of Stochastic OnsagerNet is provided `./reproduction/`.

#### Training

*(Typical install time: **XYZ**)*

Download data files `**XYZ**` from https://doi.org/10.7910/DVN/NRIX7Y and save them into the `./data` directory.

Train the model by running the Jupyter notebook `/reproduction/**XYZ**.ipynb`

The model checkpoints are saved to `./reproduction/checkpoints/`

#### Inference

*(Typical install time: **XYZ**)*

To produce main results presented in the paper, run the Jupyter notebook `./reproduction/**XYZ**.ipynb`.

For reproducibility, the default loads pre-trained checkpoints in `./reproduction/saved_checkpoints/`. Alternatvely, you may load your trained checkpoints in `./reproduction/checkpoints/`, which may result in slight statistical variations in the results.
