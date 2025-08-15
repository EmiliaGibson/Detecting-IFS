# Detecting-IFS

This repository contains code written for investigations into the detection and separation of IFS from data. 

## Usage 
This repository is structured as follows:

```
.
├── Data
│   ├── Curvilinear-Sierpinski-data.npy
│   ├── Henon-data.npy
│   ├── Logistic-data.npy
│   └── Sierpinski-data.npy
├── Utils
│   ├── utils_surfaces.py
│   ├── utils_mc.py
│   └── utils_hdi_ifs.npy
├── README.md
└── examples.py

```
### Data
This folder contains the datasets of the following examples which were used: Curvilinear Sierpinski, Sierpinski and Henon. An additional example containing a faimily of logistic maps is also included.

### Utils
This folder contains utility functions for each of the different stages.
- utils_surfaces.py has helper functions for separating different manifolds.
- utils_mc.py implements Algorithm 1.
- utils_hdi_ifs learns an analytic model representation using hidden dynamic inference.

### Examples
`examples.py` can be run to reproduce the numerical examples given in section 5.

