import scipy
import astropy
from astropy.stats import kuiper_two


def multiple_correlations(model_score, human_score):
    pearson_score = scipy.stats.pearsonr(model_score, human_score)
    spearmans_r = scipy.stats.spearmanr(model_score, human_score) # Spearman's rho
    kendalltau_r = scipy.stats.kendalltau(model_score, human_score)  # Kendall's tau
    kruskal_r = scipy.stats.kruskal(model_score, human_score)  # Gamma de Kruskal et Goodman test
    pointbiserialr = scipy.stats.pointbiserialr(model_score, human_score) # Biserial correlation coefficient
    kuiper_r = astropy.stats.kuiper_two(model_score, human_score) # Kuiper's test
    #somersd = scipy.stats.somersd(moverscores, human_score) 
    return pearson_score, spearmans_r, kendalltau_r, kruskal_r, kuiper_r #somersd 
