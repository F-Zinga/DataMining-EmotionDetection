import spacy as sp


def tokenize(sentences):

    nlp = sp.load("en_core_web_sm", exclude=["ner", "parser", "textcat"])
    nlp.add_pipe('emoji', first=True)
    docs = nlp.pipe(sentences)

    return docs
