# AI-Generated Text Classifier

**Coursework · Natural Language Processing**

Classifies text as **AI-generated** or **human-written** using classic NLP: Bag-of-Words features with Multinomial Naive Bayes and Logistic Regression.

## Approach

1. **Preprocessing** — lowercasing, URL/HTML stripping, punctuation & digit removal, tokenization, stopword removal, lemmatization (NLTK)
2. **Vectorization** — `CountVectorizer` (Bag-of-Words) on an 80/20 train/test split
3. **Models** — Multinomial Naive Bayes and Logistic Regression, trained and compared side by side with confusion matrices
4. **Inference** — a `predict_text_category()` function that preprocesses, vectorizes, and classifies new, unseen text

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Multinomial Naive Bayes | 99.66% | 100% | 95.00% | 97.44% |
| Logistic Regression | 99.66% | 100% | 95.00% | 97.44% |

Both models tied on the held-out test set. Testing the trained model on genuinely new example texts (outside the training distribution) surfaced an honest generalization limitation — a useful reminder that Bag-of-Words ignores word order/context, and that near-perfect test-set metrics don't always guarantee robust real-world performance.

## Dataset

[AI Generated Essays Dataset](https://www.kaggle.com/datasets/denvermagtibay/ai-generated-essays-dataset) (Kaggle)

## Tech stack

Python · NLTK · scikit-learn (Naive Bayes, Logistic Regression, CountVectorizer) · pandas · matplotlib · seaborn

## Notebook

`NLPFinale2lmafroodLASTT_2.ipynb` — full pipeline: preprocessing, vectorization, model training/evaluation, comparison visualizations, and a reusable prediction function.
