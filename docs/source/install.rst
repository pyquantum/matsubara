############
Installation
############

`QuTiP <http://www.qutip.org/>`_ is required to run the code. Also the standard python scientific computing packakges (numpy, scipy, cython, matplotlib) are necessary. If you have a working qutip installation then you can directly install the code in development mode by downloading the zipped version of the code and performing an "in-place" installation from your terminal using python. If you have git then you can use the git clone command, otherwise just click on the clone/download button on the Github page and get the zipped version::

    git clone https://github.com/pyquantum/matsubara.git
    python matsubara/setup.py develop

The "in-place" installation means that everything gets installed from the local folder you downloaded and you can make changes to the code which will be immediately reflected system-wide, e.g., if you insert a print statement in some part of the code and then run any example, you can see it immediately. We hope this will allow users to change things and develop the code further. Please open a pull request if you want to add some features or find a bug.

Numpy, Scipy and QuTiP are required. Install them with conda if you do not already have them using::

   conda install -c conda-forge numpy scipy qutip

Matplotlib is required for plotting and visualizations.
