#!pip install "git+https://github.com/AIPHES/emnlp19-moverscore"
#!pip install pyemd


from moverscore_v2 import get_idf_dict, word_mover_score # moverscore_v2 uses DistilBERT by default
from collections import defaultdict


def moverscore(references, candidates, human_score):
    """
    Compute the MoverScore of a list of candidate sentences with respect to one or several reference sentences.

    Parameters:
        references: list of reference sentences
        candidates: list of candidate sentences
        human_score: list of human similarity scores for each candidate sentence
    returns:
        moverscores : list of MoverScore scores for each candidate sentence
    """
    idf_dict_hyp = get_idf_dict(candidates)
    idf_dict_ref = get_idf_dict(references)

    moverscores = word_mover_score(references, candidates, idf_dict_ref, idf_dict_hyp) 
    
    return moverscores

