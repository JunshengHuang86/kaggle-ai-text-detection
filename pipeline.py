import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pickle
import re


def two_char_word_tokenizer(text):
    # Split text into words based on whitespace and remove punctuation
    words = re.findall(r'\b\w+\b', text.lower())
    two_char_tokens = [word[i:i + 2] for word in words for i in range(len(word) - 1)]
    return two_char_tokens


def main():
    file_name = 'train_drcat_04.csv'  # essay_id, text, label, source, prompt, fold
    df = pd.read_csv(file_name, index_col='essay_id')
    X = df['text']
    y = df['label']
    word_vectorizer = TfidfVectorizer(analyzer='word')
    char_vectorizer = TfidfVectorizer(analyzer='word', tokenizer=two_char_word_tokenizer)
    combined_features = FeatureUnion(
        [
            ('word_features', word_vectorizer),
            ('char_features', char_vectorizer)
        ]
    )
    pipeline = Pipeline(
        [
            ('tfidf', combined_features),
            ('classifier', LogisticRegression())
        ]
    )
    param_grid = {
        'tfidf__word_features__max_df': [0.5, 0.75, 1.0],
        'tfidf__word_features__min_df': [0, 0.01, 0.1],
        'tfidf__char_features__max_df': [0.5, 0.75, 1.0],
        'tfidf__char_features__min_df': [0, 0.01, 0.1],
        'classifier__C': [0.1, 1, 10, 100]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=1, n_jobs=-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict_proba(X_test)[:, 1]

    roc_score = roc_auc_score(y_test, y_pred)
    print("ROC AUC Score:", roc_score)
    with open('model_reg5_gs.pkl', 'wb') as file:
        pickle.dump(grid_search, file)


if __name__ == '__main__':
    main()
