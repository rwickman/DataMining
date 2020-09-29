# C-2. K-means Clustering (with Dimension Reduction)
## What Is This?
This preforms K-means Clustering with entropy-based subspace selection method.
The format this program expects can be found in data/A2-small-test.dat. 

## How to Run
*NOTE: I have only ran this in a Linux environment (Ubuntu 19.04 to be exact) with g++ version 9.2.1 so keep that in mind.*

1. Clone this entire repository.
2. Run the following command to compile the C++ code:

    ```shell
    g++ -o c main.cpp kmeans.cpp entropy_subspace.cpp
    ```
3. Finally, run the following command to run the program:

    ```shell
    ./c
    ```

## Where to Find Output
The output is saved to test.res.

## Change Settings
You can configure the settings by updating the const values at the top of main.cpp. 