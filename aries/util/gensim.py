import itertools
import logging

import gensim

logger = logging.getLogger(__name__)


def stem_tokens(tokens):
    return list(map(gensim.parsing.preprocessing.stem, tokens))


class InMemoryTextCorpus(gensim.corpora.textcorpus.TextCorpus):
    def __init__(self, texts, dictionary=None, **kwargs):
        self.texts = texts
        if "token_filters" not in kwargs:
            kwargs["token_filters"] = [stem_tokens]
        if "character_filters" not in kwargs:
            kwargs["character_filters"] = [
                gensim.parsing.preprocessing.lower_to_unicode,
                gensim.utils.deaccent,
                gensim.parsing.preprocessing.strip_multiple_whitespaces,
                gensim.parsing.preprocessing.strip_punctuation,
            ]
        super().__init__(dictionary=dictionary, **kwargs)
        # self.token_filters = [gensim.parsing.preprocessing.remove_short_tokens, gensim.parsing.preprocessing.remove_stopword_tokens]

    def __getitem__(self, item):
        return self.dictionary.doc2bow(self.preprocess_text(self.texts[item]))

    def init_dictionary(self, dictionary):
        self.dictionary = dictionary if dictionary is not None else gensim.corpora.Dictionary()
        if dictionary is None:
            logger.debug("Initializing dictionary")
            metadata_setting = self.metadata
            self.metadata = False
            self.dictionary.add_documents(self.get_texts())
            self.metadata = metadata_setting
        else:
            logger.debug("Dictionary already initialized")

    def get_texts(self):
        return list(map(self.preprocess_text, self.texts))

    def __len__(self):
        return len(self.texts)
