from bary_score import BaryScoreMetric

def Bary_score_metric(ref,hypothesis):
    '''
    Compute the Bary score of a list of candidate sentences with respect to one or several reference sentences.
    
    Parameters:
        references: list of reference sentences
        candidates: list of candidate sentences
        
    returns:
        score : list of Bary scores for each candidate sentence'''


    metric_call = BaryScoreMetric(use_idfs=False)
    metric_call.prepare_idfs(ref, hypothesis)
    return metric_call.evaluate_batch(ref, hypothesis)

