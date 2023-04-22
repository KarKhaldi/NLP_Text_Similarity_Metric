from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def BERT_sentence_transformer(list_sentences1,list_sentences2, model = SentenceTransformer('bert-base-nli-mean-tokens')):
    """
    Compute the cosine similarity of a list of candidate sentences with respect to one or several reference sentences.

    Parameters:
        list_sentences1: list of reference sentences
        list_sentences2: list of candidate sentences
        
    returns:
        bert_cos : list of cosine similarity scores for each candidate sentence
    """
    distances = []
    for i,question in enumerate(list_sentences1): 
        embeddings1 = model.encode([question])
        embeddings2 = model.encode([list_sentences2[i]])
        similarity = cosine_similarity(embeddings1, embeddings2)
        distances.append(similarity)
    bert_cos = [value[0][0] for value in distances]
    return bert_cos