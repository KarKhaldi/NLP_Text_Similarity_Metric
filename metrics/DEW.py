
from sentence_transformers import SentenceTransformer
from fastdtw import fastdtw
import numpy as np
def DEW_metric(references, candidates):
    """

    Compute the DEW - Dynamic Embedding Wrapping - score of a list of candidate sentences with respect to one or several reference sentences.
    (based on DTW - Dynmic Time Wrapping - method usually used for time series similarity comparison).
    Parameters:
        references: list of reference sentences
        candidates: list of candidate sentences
    returns:
        score : list of DEW scores for each candidate sentence
    """

    model = SentenceTransformer('bert-base-nli-mean-tokens')

    score = []
    for i in range(len(references)):
        embeddings1 = model.encode(references[i])
        embeddings2 = model.encode(candidates[i])
        distance, _ = fastdtw(embeddings1, embeddings2)
        score.append(distance)
    return -(np.array(score) - max(score))