from torchmetrics.text.infolm import InfoLM
def infoLM(references, candidates):
    """
    Compute the InfoLM score of a list of candidate sentences with respect to one or several reference sentences.

    Parameters:
        references: list of reference sentences
        candidates: list of candidate sentences
    returns: 
        score : list of InfoLM scores for each candidate sentence
    """

    infolm = InfoLM('google/bert_uncased_L-2_H-128_A-2', idf=False)
    score = []
    for i in range(len(references)):
        print(i)
        score.append(infolm(references[i], candidates[i]))
    return score