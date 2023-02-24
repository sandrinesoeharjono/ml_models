# Neural Networks

## Author
Sandrine Soeharjono (sandrinesoeharjono@hotmail.com), 2023.

## Objective
Manipulate gene expression datasets to use into various Neural Network (NN) models for binary & multi-label classification.

## Datasets 
### Stored in in `data` folder
- **droso_breeding_*.npy**: Droso-breeding dataset described in the following article and are publicly available [here](https://github.com/soham0209/Gene-Expression). The gene expression dataset is an NumPy nd-array whose rows are cohorts and columns are gene-expression values. The labels are assigned as the following: 0 to control, 1 to the Drosophilas bred on Aspergillus nidulans mutant laeA, and 2 to both the Drosophilas bred on wild Aspergillusnidulans and sterigmatocystin.
  - Publication: Dey, T.K., Mandal, S. & Mukherjee, S. Gene expression data classification using topology and machine learning models. BMC Bioinformatics 22 (Suppl 10), 627 (2021). https://doi.org/10.1186/s12859-022-04704-z

## Repo structure
```
├── environment.yml   
├── README.md   
├── data   
│  ├── Droso_breeding_labels.npy   
│  └── Droso_breeding_genex.npy   
└── src   
    ├── binary.py   
    └── multiclass.py   
```

## Setting up the environment
  1. Install miniconda by following the instructions [here](https://python-poetry.org/docs/#installation).
  2. Install the environment:  
    `conda env create` 
  3. Activate the environment:  
    `conda activate neural_network` 
  4. Run the script of interest:
    `python src/binary.py`