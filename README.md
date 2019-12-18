# OptConquest
Permutational flowshop problem solution for the OptContest 2019.

How to run:

1. Clone this repository
    ```bash
    $ git clone https://github.com/JoseBarbero/OptConquest.git
    ```
1. Go to OptConquest folder
    ```bash
    $ cd OptConquest
    ```
1. Create a conda environment
    ```bash
    $ conda env create -f optconquestenv.yml 
    ```
1. Activate conda environment
    ```bash
    $ conda activate optconquest
    ```
1. Compile Cython extension
    ```bash
    $ python setup.py build_ext --inplace
    ``` 
1. Run OptConquest
    ```bash
    $ python OptContest.py path/to/your/file
    ```

Author: Jose A. Barbero 
Contact: jabarbero@ubu.es
