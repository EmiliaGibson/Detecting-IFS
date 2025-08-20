# Detecting-IFS

This repository contains code written for the numerical experiments found in the following article: 
_Learning iterated function systems from time series of partial observations._
Developed using python 3.11.6.

## Usage 
This repository is structured as follows:

```
.
├── Data
│   ├── Curvilinear-Sierpinski-data.npy
│   ├── Henon-data.npy
├── Utils
│   ├── utils_surfaces.py
│   ├── utils_mc.py
│   └── utils_hdi_ifs.npy
├── README.md
├── examples.py
└── requirements.txd

```
### Data
This folder contains the datasets of the following examples which were used: Curvilinear Sierpinski, Sierpinski and Henon. An additional example containing a faimily of logistic maps is also included.

### Utils
This folder contains utility functions for each of the different stages.
- `utils_surfaces.py` has helper functions for separating different manifolds.
- `utils_mc.py` implements Algorithm 1.
- `utils_hdi_ifs` learns an analytic model representation using hidden dynamic inference.

### Examples
`examples.py` can be run to reproduce the numerical examples given in section 5.
Set `key = 0` for Curvlinear Sierpinski IFS (default) or `key = 1` for Henon IFS.

### Requirements
`requirements.txt` lists the required python packages and versions used in this repository.


