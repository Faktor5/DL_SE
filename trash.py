# print(check.get_words_by_prefix("sch"))
    # result_test = check.article_info(1)
    # print(result_test['title'])
    # print(result_test['url'])
    # print( {k: result_test['vector'][k] for k in list(result_test['vector'])[:10]} )

def article_info(index):
    global articles, article_word_matrix, words
    title = articles["title"].iloc[index]
    url = articles["url"].iloc[index]
    vector = article_word_matrix.iloc[index]
    # put every element of allWordVector as a key in a dictionary
    # and the corresponding value is the value of the vector variable
    # at the same index
    return {
        "title": title,
        "url": url,
        "vector": dict(list(map(
            lambda d: (d[0], round(d[1]*100)),
            list(filter(
                lambda d: d[1] > 0,
                zip(words, vector))))))
    }

def article_text_by_name(name):
    global articles
    return articles[articles["title"] == name]["article"].iloc[0]

def get_words_by_prefix(prefix):
    global words
    return list(filter(lambda word: word.startswith(prefix), words))

def article_text_by_index(index):
    global articles
    return articles["article"].iloc[index]

def round_percent(results):
    return { k:round(v,2) for k,v in results.items()}

def top_filter(results):
    return { k:v for k,v in results.items() if v > 0.0}

def search(query):
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
    v_query = m_query.toarray().reshape(check.get_shape_word_article_matrix()[0],)
    print(check.get_shape_word_article_matrix()[0],check.get_shape_word_article_matrix()[1])
    print(v_query.shape)
    print(f"Comparable Vector {v_query}")
    
    c_matrix = { v[0]:np.dot(v[1], v_query) / np.linalg.norm(v[1]) * np.linalg.norm(v_query) for v in zip(check.article_names(),article_word_matrix.values)}
    
    print(f"Comparison Matrix {c_matrix}")
    cs_matrix = dict(sorted(c_matrix.items(), key=lambda x: x[1], reverse=True))
    print(f"Sorted Comparison Matrix {cs_matrix}")
    return cs_matrix
