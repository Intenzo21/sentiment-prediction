import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from collections import Counter

# NLTK is very useful for natural language applications
import nltk

# This will be used to tokenize sentences
from nltk.tokenize.toktok import ToktokTokenizer

# This dictionary will be used to expand contractions (e.g. we'll -> we will)
from contractions import contractions_dict
import re

# Unicodedata will be used to remove accented characters
import unicodedata

# BeautifulSoup will be used to remove html tags
from bs4 import BeautifulSoup

### Lematization
import spacy
nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


model = pickle.load(open("sentiment_lr.pkl", "rb"))

vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


@app.route('/predict', methods=['POST'])
def predict():

    def strip_html_tags(text):
        """Remove html tags from text.
        """
        soup = BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text()
        return stripped_text

    def remove_accented_chars(text):
        """Remove accented characters.
        """
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    def remove_special_characters(text):
        """Remove special characters.
        """
        text = re.sub('[^a-zA-z0-9\s]', '', text)
        return text

    def lemmatize_text(text):
        """Lemmatise the text.
        """
        text = nlp(text)
        return ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])

    def expand_contractions(text, contraction_mapping=contractions_dict):
        """Find and expand text contractions
        """
        contractions_pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())),
                                          flags=re.IGNORECASE | re.DOTALL)

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(
                match) else contraction_mapping.get(match.lower())
            return first_char + expanded_contraction[1:] if expanded_contraction != None else match

        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
        return expanded_text

    # nltk.download('stopwords')
    stopword_list = nltk.corpus.stopwords.words('english')
    stopword_list.remove('no')
    stopword_list.remove('not')

    def remove_stopwords(text, is_lower_case=False):
        tokenizer = ToktokTokenizer()
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]

        if is_lower_case:
            filtered_tokens = [token for token in tokens if token not in stopword_list]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text

    def normalize_corpus(corpus, html_stripping=True, contraction_expansion=True,
                         accented_char_removal=True, text_lower_case=True,
                         text_lemmatization=True, special_char_removal=True,
                         stopword_removal=True):

        doc = corpus

        # strip HTML
        if html_stripping:
            doc = strip_html_tags(doc)

        # remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)

        # expand contractions
        if contraction_expansion:
            doc = expand_contractions(doc)

        # Lowercase the text
        if text_lower_case:
            doc = doc.lower()

        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ', doc)

        # insert spaces between special characters to isolate them
        special_char_pattern = re.compile(r'([{.(-)!}])')
        doc = special_char_pattern.sub(" \\1 ", doc)

        # lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)

        # remove special characters
        if special_char_removal:
            doc = remove_special_characters(doc)

        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)

        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)


        return doc

    str_features = " ".join([str(word) for word in request.form.values()])
    str_input = normalize_corpus(str_features)
    vectorized_input = vectorizer.transform([str_input])
    prediction = model.predict(vectorized_input)

    return render_template('index.html',
                           prediction_text=("The review is negative." if (prediction == 0) else "The review is positive."))

if __name__ == "__main__":
    app.run(debug=True)
