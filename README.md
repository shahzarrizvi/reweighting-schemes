# Learning Likelihood Ratios with Neural Network Classifiers

This is the companion code for https://arxiv.org/abs/2305.10500. 

### Quickstart 
This package uses Python 3.8 and Keras 2.9. 
<!--- Still need to figure out the tensorflow, pytorch, and nflows packages. --->

<!--- Create an environment file, like this one! 
Make sure you have `conda` installed on your system. 
```sh
conda env create -n gaia -f requirements.yml # can also use requirements_mac.yml
conda activate gaia
python -m ipykernel install --user --name gaia --display-name "gaia"
jupyter lab
```
Then, navigate to one of the notebooks in the `notebooks` folder (making sure to specify `gaia` as your kernel). --->

### Repository structure 
```sh
utils
├── losses.py    # Keras implementations of the loss functionals.
├── plotting.py  # Utility functions for plotting.
└── training.py  # Utility functions for training the classifiers.
demo.ipynb       # Demo notebook for implementing and running a simple example.
make_plots.ipynb # Shows how to replicate each of the figures in the paper.
make-data.ipynb  # Code for generating the simulated datasets used in the paper.
```

### Datasets 
The simulated Gaussian, gamma, and beta datasets are all generated using the `make-data.ipynb` notebook. The physics dataset is available [here](https://zenodo.org/records/3548091).

### Further reading: 
- ???