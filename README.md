# Detecting-IFS

This repository contains code written for investigations into the detection and separation of IFS from data. This research was part of an MSci thesis in Mathematics.

## Usage 
This repository is structured as follows:

```
.
├── Data
│   ├── Curvilinear-Sierpinski-data.npy
│   ├── Henon-data.npy
│   └── Logistic-data.npy
│   └── Sierpinski-data.npy
├── README.md
├── examples.py
├── requirements.txt
├── utils.py
```
### Data
This folder contains the datasets which were used Curvilinear Sierpinski, Sierpinski and Henon. An additional example containng a faimily of logistic maps is also included.

### utils
`utils.py` contains the function `find_surfaces` which implements Algorithm 1 in python.

### Examples
`Examples.py` can be run to reproduce Figures 1, 2 and 3

### Requirements
`requirements.txt` lists the versions of the python packages used in this repository and can be used to create a virtual environment.
