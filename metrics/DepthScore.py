from depth_score import DepthScoreMetric


def Depth_score_metric(question1,question2, model_name = 'distilbert-base-uncased'):
    """
    Compute the Depth score of a list of candidate sentences with respect to one or several reference sentences.
    
    Parameters:
        references: list of reference sentences
        candidates: list of candidate sentences
    
    returns:
        score : list of Depth scores for each candidate sentence
    """
    metric_call = DepthScoreMetric(model_name, layers_to_consider=4)
    final_preds = metric_call.evaluate_batch(question1, question2)
    return final_preds["depth_score"]


