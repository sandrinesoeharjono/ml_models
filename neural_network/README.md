# Image Classification

## Author
Sandrine Soeharjono (sandrinesoeharjono@hotmail.com), 2023.

## Objective
Manipulate gene expression data of _ samples in Python.

## Dataset
The dataset was submitted to NCBI on Oct 7, 2021 by María José Jiménez (mjjimenez@cnio.es) and is publicly available at the following link: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE185513

## Repo structure
├── environment.yml
├── README.md
├── data
│   ├── GSE185513_ALLsamples.normalizedCounts.txt
│   └── TODO
├── preprocessing.py
└── model.py

## Setting up the environment
  1. Install miniconda by following the instructions [here](https://python-poetry.org/docs/#installation).
  2. Install the environment:  
    `conda env create` 
  3. Activate the environment:  
    `conda activate neural_network` 
  4. Run the script:
    `python model.py`