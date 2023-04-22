import mauve


def MAUVE_metric_single_sentence(question1,question2):
    """
    
    Compute the MAUVE score of a candidate sentence with respect to a reference sentence.

    Parameters:
        question1: reference sentence
        question2: candidate sentence

    returns:
        score : MAUVE score
    """
    out = mauve.compute_mauve(p_text=[question1], q_text=[question2])
    return out.mauve

def MAUVE_metric(references, candidates):
    """
    Compute the MAUVE score of a list of candidate sentences with respect to one or several reference sentences.

    Parameters:
        references: list of reference sentences
        candidates: list of candidate sentences

    returns:
        score : list of MAUVE scores for each candidate sentence
    """
    score = []
    for i in range(len(references)):
        question1 = references[i]
        question2 = candidates[i]
        score.append(MAUVE_metric(question1,question2)) 
    return score