# Import packages
import os
import re
import nltk
import string
import pandas as pd
from typing import List, Tuple, Any


def ReviewSentimentMatch(path: str) -> pd.DataFrame:
    """
        Match review with its label (folder name)

        Parameters
        ----------
        - path: str 
                path to folder with label folders with reviews in them

        Returns
        -------
        pd.DataFrame {index: int, review: str, sentiment: str}
    """

    dataset = []
    for label in os.listdir(path):
        if label.find('.') == -1:
            labelpath = path + label + '\\'
            for file in os.listdir(labelpath):
                with open(labelpath + file, "r", encoding='utf-8') as review:
                    dataset.append((review.read(), label))

    return pd.DataFrame(dataset, columns=['review', 'sentiment'])


def RemoveHTML(text: str) -> str:
    """
        Removes all html tags from text

        Parameters
        ----------
        - text : str
                 text to remove tags from

        Returns
        -------
        str {text where html tags were removed}
    """
    rule = re.compile(r'<.*?>')
    return re.sub(rule, '', text)


def RemoveSWSC(text: str, stopwords: List[str]) -> List[str]:
    """
        Removes stopwords and special characters from text

        Parameters
        ----------
        - text : str
                 text to stopwords and special characters from
        - stopwords : List[str]
                      list of stopwords of the required language

        Returns
        -------
        List[str] {list of words of original text where stopwords and special
                   characters were removed}
    """
    return [word.lower() for word in nltk.word_tokenize(text)
            if (word not in string.punctuation)
            and (word.lower() not in stopwords)]


def StemText(text: List[str], stemmer: nltk.SnowballStemmer) -> str:
    """
        Stem words in text

        Parameters
        ----------
        - text : List[str]
                 list of words of original text to stem

        Returns
        -------
        str {text where each word was stemmed}
    """
    return ' '.join([stemmer.stem(word) for word in text])


def ConstructRSWSC(stopwords: List[str]):
    """
        Construct RemoveSWSC lambda function with given stopwords

        Parameters
        ----------
        - stopwords : List[str]
                      list of stopwords of the required language

        Returns
        -------
        lambda function {RemoveSWSC lambda function with given stopwords}
    """
    return lambda x: RemoveSWSC(x, stopwords)


def ConstructST(stemmer: nltk.SnowballStemmer):
    """
        Construct StemText lambda function with given stemmer

        Parameters
        ----------
        - stemmer : nltk.SnowballStemmer
                    stemmer for given language

        Returns
        -------
        lambda function {StemText lambda function with given stemmer}
    """
    return lambda x: StemText(x, stemmer)


def PrepareData(data_: pd.DataFrame,
                language: str,
                encoder: Tuple[List[Any], List[Any]]
                = (['bad', 'neutral', 'good'], [-1, 0, 1]))\
        -> pd.DataFrame:
    """
        Encode data labels.
        Remove html tags, stopwords and special characters from reviews.
        Stem word in reviews.

        Parameters
        ----------
        - data_ : pd.DataFrame
                  original dataset  {review: str, sentiment: str}
        - language : str
                     language of reviews
        - encoder : Tuple[List[Any], List[Any]]
                    First list of tuple is original labels, second - encoded ones

        Returns
        -------
        pd.DataFrame {reviews: str, sentiment: str}
    """
    # Encode labels
    data = data_.copy()
    data.sentiment = data.sentiment.replace(*encoder)

    # Remove html tags
    data.review = data.review.apply(RemoveHTML)

    # Remove stopwords and special characters
    stopwords = nltk.corpus.stopwords.words(language)
    data.review = data.review.apply(ConstructRSWSC(stopwords))

    # Stem words
    stemmer = nltk.SnowballStemmer(language)
    data.review = data.review.apply(ConstructST(stemmer))

    return data


def FeedReview(text: str, language: str) -> str:
    """
        Remove html tags, stopwords and special characters from review.
        Stem word in review.

        Parameters
        ----------
        - text : str
                 review
        - language : str
                     language of the review

        Returns
        -------
        str {prepared review}
    """
    # Remove html tags
    text = RemoveHTML(text)

    # Remove stopwords and special characters
    stopwords = nltk.corpus.stopwords.words(language)
    text = RemoveSWSC(text, stopwords)
    
    # Stem words
    stemmer = nltk.SnowballStemmer(language)
    text = StemText(text, stemmer)

    return text
