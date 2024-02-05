# ica-tutorial

# Introduction
Tutorial on Blind Source Separation (BSS) and Independent Component Analysis (ICA).

The repository is structured as follows:

- the [`Tutorial on ICA.ipynb`](https://github.com/nihil21/ica-tutorial/blob/main/Tutorial%20on%20ICA.ipynb) Jupyter notebook contains the actual tutorial;
- the [`ica_tutorial`](https://github.com/nihil21/ica-tutorial/blob/main/ica_tutorial) folder contains the utility code used in the notebook:
    - the [`ica`](https://github.com/nihil21/ica-tutorial/blob/main/ica_tutorial/ica) package contains the implementation of ICA algorithms (i.e., FastICA and EFICA); 
    - the [`preprocessing`](https://github.com/nihil21/ica-tutorial/blob/main/ica_tutorial/preprocessing) package contains common preprocessing steps (e.g., filtering and whitening);
    - the [`rls`](https://github.com/nihil21/ica-tutorial/blob/main/ica_tutorial/rls) package contains the implementation of Recursive Least Squares (RLS) algorithms for ICA and whitneing;
    - the [`plotting`](https://github.com/nihil21/ica-tutorial/blob/main/ica_tutorial/plotting) package contains functions for plotting;
    - the [`utils`](https://github.com/nihil21/ica-tutorial/blob/main/ica_tutorial/utils) package contains utility functions.

## Environment setup
The code is compatible with Python 3.7+. To create and activate the Python environment, run the following commands:
```
python -m venv <ENV_NAME>
source <ENV_NAME>/bin/activate
```

Then, **from within the virtual environment**, the required packages can be installed with the following command:
```
pip install -r requirements.txt
```

## License
All files are released under the Apache-2.0 license (see [`LICENSE`](https://github.com/pulp-bio/unibo-tbiocas24-dataset/blob/main/LICENSE)).

## TO-DOs
- [ ] complete tutorial with non-stationary conditions