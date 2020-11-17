# C-4. Text & Graph Mining: Implement PageRank on Covid-19 Papers
## What Is This?
This program will create a graph were papers will be nodes and edges will be created based on their TF-IDF cosine similarity. PageRank will be computed on the created graph and the top ranking titles will be written to a file. 

## How to Run

1. Clone this entire repository.

2. Install dependencies (you may need to use pip not pip3 depending on OS):
    ```shell
    pip3 install numpy scikit-learn tqdm nltk 
    ```

3. Run the following command to run the Python3 code:

    ```shell
    python3 main.py --input <path/to/input.json>
    ```

    If on Windows, you may need to run this instead:
    
    ```shell
    python main.py --input <path/to/input.json>
    ```

## Change Settings
You can configure the settings by passing CLI arguments specified in main.py.

For example, changing the input and output file:

```shell
python3 main.py --input DMPaper.json --output DMPaper_output.txt
```