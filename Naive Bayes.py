# Naive Bayes Code for Machine Learning 

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.datasets import make_classification, fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# Gaussian NB (continuous features)
X, y = make_classification(n_samples=1000, n_features=10, random_state=46)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46)

gnb = GaussianNB(var_smoothing=1e-9)
gnb.fit(X_train, y_train)
print("GaussianNB Accuracy:", gnb.score(X_test, y_test))

# Multinomial NB (text/discrete counts)
categories = ['alt.atheism', 'sci.space', 'comp.graphics']
newsgroups = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_text = vectorizer.fit_transform(newsgroups.data)
y_text = newsgroups.target

mnb = MultinomialNB(alpha=1.0)  # Laplace smoothing
mnb.fit(X_text, y_text)
print("MultinomialNB Accuracy:", mnb.score(X_text, y_text))

# Bernoulli NB (binary features)
bnb = BernoulliNB(alpha=1.0)
X_binary = (X > X.mean(axis=0)).astype(int)
bnb.fit(X_binary, y)
print("BernoulliNB Accuracy:", bnb.score(X_binary, y))
