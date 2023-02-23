# Image Classification

## Author
Sandrine Soeharjono (sandrinesoeharjono@hotmail.com), 2023.

## Objective
Build an image classification model using the [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/) libraries in Python to differentiate between two groups: cats and dogs.

## Dataset
Publicly available at the following link: https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip

## Repo structure
├── environment.yml
├── README.md
├── data
│   ├── CDLA-Permissive-2.0.pdf
│   ├── pet_images
│         ├── cat
│         └── dog
├── data.py
└── model.py

## Setting up the environment
  1. Install miniconda by following the instructions [here](https://python-poetry.org/docs/#installation).
  2. Install the environment:  
    `conda env create` 
  3. Activate the environment:  
    `conda activate image_classification` 
  4. Run the script:
    `python model.py`