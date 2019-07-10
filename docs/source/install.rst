############
Installation
############

`QuTiP <http://www.qutip.org/>`_ is required to run the code. Also the standard python scientific computing packakges (numpy, scipy, cython, matplotlib) are necessary. Download the zipped version of the code, unzip it and install using the following command from your terminal from the matsubara folder::

    python setup.py develop
    
This performs an "in-place" installation which means that everything gets installed from the local folder you downloaded and you can make changes to the code which will be immediately reflected system-wide, e.g., if you insert a print statement in some part of the code and then run any example, you can see it immediately. We hope this will allow users to change things and develop the code further. Please open a pull request if you want to add some features or find a bug.

Numpy, Scipy and QuTiP are required. Install them with conda/pip if you do not already have them using::

   pip install cython numpy scipy
   pip install qutip

Matplotlib is required for plotting and visualizations.
