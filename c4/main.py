import json, re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from scipy.sparse import csr_matrix
import numpy as np


ALPHA = 0.15
PR_ITER = 20

stemmer = PorterStemmer()
def clean_text(text):
    
    # Make all the words lowercasse
    text = text.lower()

    # Replace newlines and tabs with spaces
    newline_regex = re.compile("[\n\t]")
    text = newline_regex.sub(" ", text)

    # Remove all non alphabetic characters
    alphanumeric_regex = re.compile("[^ a-z0-9]")
    text = alphanumeric_regex.sub("", text)

    # Perform stemming on every word
    stemmed_words = []
    for word in text.split():
        stemmed_words.append(word)
    
    return stemmed_words


def top_pr(proximity_matrix, cosine_threshold):
    num_nodes = proximity_matrix.shape[0]

    # Create sparse-encoding of the adjacency matrix
    adj_matrix = csr_matrix(proximity_matrix.shape)
    pr_scores = np.repeat(1, num_nodes)
    max_sim = 0
    for i in range(num_nodes):
        for j, sim in enumerate(proximity_matrix[i]):
            # Don't create self-cycles
            if i == j:
                continue
            if sim >= cosine_threshold:
                adj_matrix[i,j] = 1
                max_sim = max(sim, max_sim)

    # Get out degrees of each node
    out_degrees = adj_matrix.sum(axis=-1)

    # Prevent division by zero error 
    out_degrees[out_degrees == 0] = 1

    # Create transition probability
    M = adj_matrix.multiply(1/out_degrees)

    for i in range(PR_ITER):
        pr_scores = ALPHA * (1/num_nodes) + (1-ALPHA) * M.T.dot(pr_scores.T) 
        #print(pr_scores)

    # Get the papers with higher PR score
    rank_by_pr = np.argsort(pr_scores)
    top_10 =  rank_by_pr[:10]
    return top_10




covid_papers = "CovidPaper.json"

papers = []
with open(covid_papers) as f:
    for line in f:
        papers.append(json.loads(line))

# papers_dict = json.load(f)
# print(type(papers_dict))
corpus = []
for i in range(len(papers)):
    corpus.append(clean_text(papers[i]["abstract"]))

# Fit TF-IDF on corpus
vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
X = vectorizer.fit_transform(corpus)

# Pairwise cosine similarity on abstracts
proximity_matrix = cosine_similarity(X, X)

for threshold in [0.25, 0.5, 0.75]:
    top_10 = top_pr(proximity_matrix, threshold)
    print(top_10)