from bert_score import score

def BertScore_metric_single_sentence(question1,question2):
    """
    Compute the BertScore of a candidate sentence with respect to one reference sentence.    

    Parameters:
        question1: reference sentence
        question2: candidate sentence
    returns:
        score : BertScore for the candidate sentence
        """
    P, R, F1 = score([question1], [question2], lang='en', verbose=False)
    return R

def BertScore_metric(references, candidates):
    """

    Compute the BertScore of a list of candidate sentences with respect to one or several reference sentences.

    Parameters:
        references: list of reference sentences
        candidates: list of candidate sentences

    returns:
        score : list of BertScore scores for each candidate sentence
    """
    scoring = []
    for i in range(len(scoring),len(references)):
        question1 = references[i]
        question2 = candidates[i]
        scoring.append(BertScore_metric_single_sentence(question1,question2))
    return scoring