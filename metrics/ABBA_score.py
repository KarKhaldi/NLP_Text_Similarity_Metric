
from sentence_transformers import SentenceTransformer
import numpy as np

def DEW_metric(references, candidates,model_input = 'bert-base-nli-mean-tokens'):
    """

    Compute the ABBA_Score - Average Based BertEmbedding Approach - score of a list of candidate sentences with respect to one reference sentences.


    Parameters:
        references: list of reference sentences
        candidates: list of candidate sentences
    returns:
        score : list of DEW scores for each candidate sentence
    """

    model = SentenceTransformer(model_input)
    score = []
    for i in range(len(references)):
        embeddings1 = model.encode(references[i])
        embeddings2 = model.encode(candidates[i])
        # # Quantize the time series to integers for Levenshtein Distance calculation
        quantized_ts1 = np.round(embeddings1)
        quantized_ts2 = np.round(embeddings2)
        distance = dist_between_ts(quantized_ts1, quantized_ts2)
        score.append(distance)

    # normalize the score
    score = [1- (value/768) for value in score]
    return score





# calculate our special distance between quantized time series
def dist_between_ts(quantized_ts1, quantized_ts2):
    """
    Calculate the distance between two quantized time series.

    Args:
        quantized_ts1: quantized time series 1
        quantized_ts2: quantized time series 2

    Returns:
        distance: distance between the two time series
    """
    distance = 0
    for element in range(len(quantized_ts1)):
        # if element in ts1 equals element in ts2 then distance is 0
        if quantized_ts1[element] == quantized_ts2[element]:
            distance += 0
        elif quantized_ts1[element] == 0:
            distance += 1
        else:
            if np.sign(quantized_ts1[element]) == np.sign(quantized_ts2[element]):
                distance += 0.5
            elif quantized_ts1[element] == - quantized_ts2[element]:
                distance += 0.5
            else:
                distance += 0.75
    return distance
