from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from nltk.stem.snowball import SnowballStemmer

from utils import *

def label_stocks(row):
    if row['Close'] - row['Open'] < 0:
        return -1
    else:
        return 1

apple_headlines = ("Data/News data/apple_deduped.csv", "Data/Stock data/Stocks/aapl.us.txt")
amazon_headlines = ("Data/News data/amazon_deduped.csv", "Data/Stock data/Stocks/amzn.us.txt")
facebook_headlines = ("Data/News data/facebook_deduped.csv", "Data/Stock data/Stocks/fb.us.txt")

apple_reddit = ("Data/News data/apple_reddit.csv", "Data/Stock data/Stocks/aapl.us.txt")
amazon_reddit = ("Data/News data/amazon_reddit.csv", "Data/Stock data/Stocks/amzn.us.txt")
facebook_reddit = ("Data/News data/facebook_reddit.csv", "Data/Stock data/Stocks/fb.us.txt")


df = pd.read_csv(apple_headlines[0])
stock = pd.read_csv(apple_headlines[1])

df = pd.merge(df, stock, on="Date", how="inner")

df['value'] = df.apply(lambda row: label_stocks(row), axis=1)

df = df[['text', 'value']]

pos = df.loc[df['value'] == 1, 'text'].copy().reset_index(drop=True)
neg = df.loc[df['value'] == -1, 'text'].copy().reset_index(drop=True)

neg = pd.concat([pd.DataFrame(neg), pd.DataFrame(np.zeros(neg.shape), columns=['class'])], 1)
pos = pd.concat([pd.DataFrame(pos), pd.DataFrame(np.ones(pos.shape), columns=['class'])], 1)

np.random.seed(42)
rand = np.random.permutation(pos.shape[0])
pos = pos.iloc[rand[:neg.shape[0]]].reset_index(drop=True)

df = pd.concat([pos, neg]).sample(frac=1).reset_index(drop=True)
df.head()

X_train, X_test, y_train, y_test = train_test_split(df['text'].values, df['class'].values, test_size=0.2,
                                                    random_state=42)

ENGLISH_STOP_WORDS = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "s", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"])


def stemmed_words(doc):
    stemmer = SnowballStemmer('english')
    analyzer = CountVectorizer().build_analyzer()
    return (stemmer.stem(w) for w in analyzer(doc))


nb_b = BernoulliNB()
nb_m = MultinomialNB()

for i in range(1, 5):
    # Add analyzer=stemmed_words as a parameter to stem each word
    vect = CountVectorizer(strip_accents='unicode', stop_words=ENGLISH_STOP_WORDS, binary=True, ngram_range=(1, i))

    tf_train = vect.fit_transform(X_train)
    tf_test = vect.transform(X_test)

    nb_b.fit(tf_train, y_train)

    nb_b.score(tf_train, y_train)

    y_pred = nb_b.predict(tf_test)

    print(f"Bernoulli NB, CountVect with {i}-gram")
    print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
    print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred) * 100))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    nb_m.fit(tf_train, y_train)

    nb_m.score(tf_train, y_train)

    y_pred = nb_m.predict(tf_test)

    print(f"Multinomial NB, CountVect with {i}-gram")
    print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
    print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred) * 100))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

for i in range(1, 5):
    vect = TfidfVectorizer(strip_accents='unicode', stop_words=ENGLISH_STOP_WORDS, binary=True, ngram_range=(1, i))

    tfidf_train = vect.fit_transform(X_train)
    tfidf_test = vect.transform(X_test)

    nb_b.fit(tfidf_train, y_train)

    nb_b.score(tfidf_train, y_train)

    y_pred = nb_b.predict(tfidf_test)

    print(f"Bernoulli NB, TfidfVect with {i}-gram")
    print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
    print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred) * 100))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    nb_m.fit(tfidf_train, y_train)

    nb_m.score(tfidf_train, y_train)

    y_pred = nb_m.predict(tfidf_test)

    print(f"Multinomial NB, TfidfVect with {i}-gram")
    print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
    print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred) * 100))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
