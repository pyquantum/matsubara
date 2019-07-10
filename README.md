# matsubara

## Modelling the ultra-strongly coupled spin-boson model with unphysical modes
Neill Lambert, Shahnawaz Ahmed, Mauro Cirio, Franco Nori

The is the code to reproduce the results in [arXiv:1903.05892](https://arxiv.org/abs/1903.05892). A special `matsubara.heom.HeomUB` class is provided to implement the Hierarchical Equations of Motion method adapted for the underdamped Brownian motion spectral density.  

We focus on the zero temperature case where the correlation function can be expressed using four exponents - two "non-Matsubara" and two "Matsubara" exponents. The "Matsubara" exponents give rise to seemingly unphysical modes which nevertheless are essential to get correct physical results, e.g., virtual photon population.  

We also present a Reaction Coordinate method and a pseudomode approach to study the problem and show the crucial role of the Matsubara terms by effectively expressing them with a biexponential fit. 

Documentation, installation instructions and several examples are available at: [https://matsubara.readthedocs.io/en/latest/](https://matsubara.readthedocs.io/en/latest/)

# Installation

[QuTiP](http://www.qutip.org/) is required to run the code. Also the standard python scientific computing packakges (numpy, scipy, cython, matplotlib) are necessary. Download the zipped version of the code, unzip it and install using the following command from your terminal from the matsubara folder

```
python setup.py develop
```
    
This performs an "in-place" installation which means that everything gets installed from the local folder you downloaded and you can make changes to the code which will be immediately reflected system-wide, e.g., if you insert a print statement in some part of the code and then run any example, you can see it immediately. We hope this will allow users to change things and develop the code further. Please open a pull request if you want to add some features or find a bug.

### QuTiP installation

QuTiP can be installed in a fresh environment using the following commands:

```
pip install cython numpy scipy
pip install qutip
```

Matplotlib is required for plotting and visualizations.

# Examples

In `matsubara/docs/source/examples/` there are several examples from the paper which can be easily
reproduced. Just run them using python from the examples folder and the corresponding plots with be available in `matsubara/docs/source/examples/plots`.

```
cd matsubara/docs/source/examples/
python example_heom.py
```

The basic calculation of the Matsubara and non Matsubara modes can be done in the following way:

```python
from matsubara.correlation import (nonmatsubara_exponents,
                                   matsubara_zero_analytical,
                                   biexp_fit, sum_of_exponentials)


import numpy as np
coup_strength, bath_broad, bath_freq = 0.2, 0.05, 1.
tlist = np.linspace(0, 100, 1000)

# Zero temperature case beta = 1/kT
beta = np.inf
ck1, vk1 = nonmatsubara_exponents(coup_strength, bath_broad, bath_freq, beta)

# Analytical zero temperature calculation of the Matsubara correlation
mats_data_zero = matsubara_zero_analytical(coup_strength, bath_broad, bath_freq, tlist)

# Fitting a biexponential function
ck20, vk20 = biexp_fit(tlist, mats_data_zero)

print("Coefficients:", ck1, ck20)
print("Frequencies:", vk1, vk20)
```

```
Coefficients: [0.         0.02000625] [-0.00020758 -0.00010652]
Frequencies: [-0.025+0.99968745j -0.025-0.99968745j] [-1.61341629 -0.3296043 ]
```
![](docs/source/examples/plots/matsfitting.png)

# Documentation

The documentation is available at [http://matsubara.readthedocs.io](http://matsubara.readthedocs.io) but you can also build your own local version by modifying the `.RST` files in `docs/source/` and running the `make html` command from the docs folder. The documentation is build using `sphinx`. Please make sure you have [sphinx](http://www.sphinx-doc.org/en/master/) installed along with the dependency [sphinxcontrib-bibtex](https://sphinxcontrib-bibtex.readthedocs.io/en/latest/) for generating the bibliography.

Install sphinx with:

```
pip install -U Sphinx
pip install sphinxcontrib-bibtex sphinx_rtd_theme
```

Generate the docs:
```
cd docs/
make html
```
