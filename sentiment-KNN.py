from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

from utils import *

df = pd.read_csv('apple_merged.csv')

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

vect = CountVectorizer(strip_accents='unicode', stop_words=ENGLISH_STOP_WORDS, binary=True)

tf_train = vect.fit_transform(X_train)
tf_test = vect.transform(X_test)

pipeline_knn = make_pipeline(vect, KNeighborsClassifier())
param_grid = {'kneighborsclassifier__n_neighbors': np.arange(1, 50)}

grid_knn = GridSearchCV(pipeline_knn,
                        param_grid,
                        cv=5,
                        scoring="roc_auc",
                        verbose=1,
                        n_jobs=-1)

grid_knn.fit(X_train, y_train)
grid_knn.score(X_test, y_test)

print("Count Vectorizer:")

print(grid_knn.best_params_)

print(grid_knn.best_score_)


def report_results(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)

    auc = roc_auc_score(y, pred_proba)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)
    result = {'auc': auc, 'f1': f1, 'acc': acc, 'precision': prec, 'recall': rec}
    return result


print(report_results(grid_knn.best_estimator_, X_test, y_test))

vect = TfidfVectorizer(strip_accents='unicode', stop_words=ENGLISH_STOP_WORDS, binary=True, ngram_range=(1, 2),
                       max_df=0.9, min_df=3, sublinear_tf=True)


tfidf_train = vect.fit_transform(X_train)
tfidf_test = vect.transform(X_test)

print("TFidf Vectorizer:")

pipeline_knn = make_pipeline(vect, KNeighborsClassifier())

grid_knn.fit(X_train, y_train)
grid_knn.score(X_test, y_test)

print(grid_knn.best_params_)

print(grid_knn.best_score_)

print(report_results(grid_knn.best_estimator_, X_test, y_test))
