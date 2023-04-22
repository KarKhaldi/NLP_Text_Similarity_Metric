import scipy


def multiple_correlations(moverscores, human_score):
    pearson_score = scipy.stats.pearsonr(moverscores, human_score)
    spearmans_r = scipy.stats.spearmanr(moverscores, human_score) # Spearman's rho
    kendalltau_r = scipy.stats.kendalltau(moverscores, human_score)  # Kendall's tau
    kruskal_r = scipy.stats.kruskal(moverscores, human_score)  # Gamma de Kruskal et Goodman test
    pointbiserialr = scipy.stats.pointbiserialr(moverscores, human_score) # Biserial correlation coefficient
    somersd = scipy.stats.somersd(moverscores, human_score) 
    return pearson_score, spearmans_r, kendalltau_r, kruskal_r, pointbiserialr, somersd 
