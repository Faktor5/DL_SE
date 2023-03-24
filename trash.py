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
