import libraries.data_cleaner as dc
import numpy as np

def round_percent(results):
    return { k:round(v,2) for k,v in results.items()}

def top_filter(results):
    return { k:v for k,v in results.items() if v > 0.0}

def search(query, model, article_word_matrix, a_filter):
    cl_query = dc.clean_text(query)
    print(f"Cleaned --{cl_query}--")
    m_query = model.transform([cl_query])
    # the vector has for every word in the corpus a value
    # which is the tfidf value of the word in the query
    # the tfidf value is the product of the tf value and the idf value
    # tf value is the frequency of the word in the query
    # idf value is the inverse document frequency of the word in the corpus
    # the tfidf value is the importance of the word in the query
    # the higher the tfidf value, the more important the word is in the query
    # the tfidf value is the same for every word in the corpus
    print(f"Vectorized {m_query}")
    v_query = m_query.toarray().reshape(a_filter.get_shape_word_article_matrix()[0],)
    print(a_filter.get_shape_word_article_matrix()[0],a_filter.get_shape_word_article_matrix()[1])
    print(v_query.shape)
    print(f"Comparable Vector {v_query}")
    
    c_matrix = { v[0]:np.dot(v[1], v_query) / np.linalg.norm(v[1]) * np.linalg.norm(v_query) for v in zip(a_filter.article_names(),article_word_matrix.values)}
    
    print(f"Comparison Matrix {c_matrix}")
    cs_matrix = dict(sorted(c_matrix.items(), key=lambda x: x[1], reverse=True))
    print(f"Sorted Comparison Matrix {cs_matrix}")
    return cs_matrix
