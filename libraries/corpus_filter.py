class CorpusFilter:
    articles = None
    words = None
    article_word_matrix = None

    def __init__(self, articles, words, article_word_matrix):
        self.articles = articles
        self.words = words
        self.article_word_matrix = article_word_matrix
        

    def article_info(self, index):
        title = self.articles["title"].iloc[index]
        url = self.articles["url"].iloc[index]
        vector = self.article_word_matrix.iloc[index]
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
                    zip(self.words, vector))))))
        }

    def article_text_by_name(self, name):
        return self.articles[self.articles["title"] == name]["article"].iloc[0]

    def get_words_by_prefix(self, prefix):
        return list(filter(lambda word: word.startswith(prefix), self.words))

    def article_text_by_index(self, index):
        return self.articles["article"].iloc[index]
