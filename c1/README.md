# C-1. Frequent Itemset Mining using FP-tree
## How to Run
*NOTE: I have only ran this in a Linux environment (Ubuntu 19.04 to be exact), so keep that in mind.*

1. Clone this entire repository.
2. Run the following command to compile the C++ code:

    ```shell
    g++ main.cpp -o c fptree.h fptree.cpp fpnode.h fpnode.cpp -lstdc++fs
    ```
3. Finally, run the following command to run the program:

    ```shell
    ./c
    ```

## Where to Find Output
The output for the topics can be found in the ./output directory.