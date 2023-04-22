from rouge import Rouge
def Rouge_metric_single_sentence(question1,question2):
    rouge = Rouge()
    scores = rouge.get_scores(question1, question2) 
    return scores[0]['rouge-2']['r']

def Rouge_metric(references, candidates):
    """
    Compute the Rouge score of a list of candidate sentences with respect to one or several reference sentences.

    Parameters:
        references: list of reference sentences
        candidates: list of candidate sentences

    returns:
        score : list of Rouge scores for each candidate sentence
    """

    score = []
    for i in range(len(references)):
        question1 = references[i]
        question2 = candidates[i]
        score.append(Rouge_metric_single_sentence(question1,question2))

    return score