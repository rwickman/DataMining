import json, re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from scipy.sparse import csr_matrix
import numpy as np
import argparse

stemmer = PorterStemmer()
def clean_text(text):
    
    # Make all the words lowercasse
    text = text.lower()

    # Replace newlines and tabs with spaces
    newline_regex = re.compile("[\n\t]")
    text = newline_regex.sub(" ", text)

    # Remove all non alphabetic characters
    alphabetic_regex = re.compile("[^ a-z]")
    text = alphabetic_regex.sub("", text)

    # Perform stemming on every word
    stemmed_words = []
    for word in text.split():
        stemmed_words.append(word)
    
    return stemmed_words


def top_pr(args, proximity_matrix, cosine_threshold):
    num_nodes = proximity_matrix.shape[0]

    # Create sparse-encoding of the adjacency matrix
    adj_matrix = csr_matrix(proximity_matrix.shape)
    pr_scores = np.repeat(1, num_nodes)
    num_edges = 0
    for i in range(num_nodes):
        for j, sim in enumerate(proximity_matrix[i]):
            # Don't create self-cycles
            if i == j:
                continue
            if sim >= cosine_threshold:
                adj_matrix[i,j] = 1
                num_edges += 1

    # Get out degrees of each node
    out_degrees = adj_matrix.sum(axis=-1)

    # Prevent division by zero error 
    out_degrees[out_degrees == 0] = 1

    # Create transition probability
    M = adj_matrix.multiply(1/out_degrees).T
    #print(M)
    for i in range(args.pr_iter):
        print(i)
        pr_scores = args.alpha * (1/num_nodes) + (1-args.alpha) * M.dot(pr_scores)
        #print(pr_scores)

    # Get the papers with higher PR score
    #rank_by_pr = np.argsort(pr_scores)
    
    
   
    return pr_scores, num_edges


def create_results(args):
    # Read in the papers
    papers = []
    with open(args.input) as f:
        for line in f:
            papers.append(json.loads(line))

    corpus = []
    paper_titles = []
    #for i in range(len(papers)):
    for i in range(len(papers)):
        corpus.append(clean_text(papers[i]["abstract"]))
        paper_titles.append(papers[i]["title"])

    # Fit TF-IDF on corpus
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    X = vectorizer.fit_transform(corpus)

    # Pairwise cosine similarity on abstracts
    proximity_matrix = cosine_similarity(X, X)

    # Compute PageRank and output results
    with open(args.output, "w") as f:
        for threshold in args.thetas:
            print(threshold)
            pr_scores, num_edges = top_pr(args, proximity_matrix, threshold)
            ranked_papers = sorted(zip(pr_scores, paper_titles), key=lambda x: (-x[0], x[1]) )
            f.write("θ = {}, |V| = {}, |E| = {},\n".format(threshold, proximity_matrix.shape[0], num_edges))
            for i, top_pair in enumerate(ranked_papers[:args.num_top]):
                f.write(top_pair[1].replace("\n", "") + ";\n")


def main(args):
    create_results(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.15,
            help="Alpha value for PageRank.")
    parser.add_argument("--pr_iter", type=int, default=20,
            help="Number of PageRank iterations.")
    parser.add_argument("--num_top", type=int, default=10,
            help="Number of top PageRank title results to write.")
    parser.add_argument("--thetas", type=float, nargs="*", default=[0.25,0.5,0.75],
            help="Cosine threshold to create an edge for graph.")
    parser.add_argument("--output", default="results.txt",
            help="Output path/filename.")
    parser.add_argument("--input", default="CovidPaper.json",
            help="Input JSON path/filename.")
    main(parser.parse_args())