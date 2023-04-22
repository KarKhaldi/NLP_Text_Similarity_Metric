from nltk.translate.bleu_score import sentence_bleu


def Bleu_metric_single_sentence(question1,question2):
    """
    
    Compute the Bleu score of a candidate sentence with respect to one reference sentence.

    Parameters:
        question1: reference sentence
        question2: candidate sentence
    returns:
        score : Bleu score for the candidate sentence
    """
    reference = [question1.split()]
    candidate = question2.split()
    score = sentence_bleu(reference, candidate)
    return score

def Bleu_metric(references, candidates):
    """
    Compute the Bleu score of a list of candidate sentences with respect to one or several reference sentences.

    Parameters:
        references: list of reference sentences
        candidates: list of candidate sentences

    returns:
        score : list of Bleu scores for each candidate sentence
    """

    score = []
    for i in range(len(references)):
        question1 = references[i]
        question2 = candidates[i]
        score.append(Bleu_metric_single_sentence(question1,question2))
    return score