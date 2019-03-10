from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from utils import *


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

vect_count = CountVectorizer(strip_accents='unicode', stop_words=ENGLISH_STOP_WORDS, binary=True)

vect_tfidf = TfidfVectorizer(strip_accents='unicode', stop_words=ENGLISH_STOP_WORDS, binary=True, ngram_range=(1, 2),
                             max_df=0.9, min_df=3, sublinear_tf=True)
search = False
optimal_C = 1.0

if search:
    # Try Count Vectorizer
    kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    pipeline_svm = make_pipeline(vect_count, SVC(probability=True, kernel="linear", class_weight="balanced"))

    grid_svm = GridSearchCV(pipeline_svm,
                            param_grid={'svc__C': [0.01, 0.1, 1]},
                            cv=kfolds,
                            scoring="roc_auc",
                            verbose=1,
                            n_jobs=-1)

    grid_svm.fit(X_train, y_train)
    grid_svm.score(X_test, y_test)

    print("Count Vectorizer:")

    print(grid_svm.best_params_)

    print(grid_svm.best_score_)

    print(report_results(grid_svm.best_estimator_, X_test, y_test))

    # Try Tfidf Vectorizer

    print("TFidf Vectorizer:")

    pipeline_svm = make_pipeline(vect_tfidf, SVC(probability=True, kernel="linear", class_weight="balanced"))

    grid_svm = GridSearchCV(pipeline_svm,
                            param_grid={'svc__C': [0.01, 0.1, 1]},
                            cv=kfolds,
                            scoring="roc_auc",
                            verbose=1,
                            n_jobs=-1)

    grid_svm.fit(X_train, y_train)
    grid_svm.score(X_test, y_test)

    print(grid_svm.best_params_)

    print(grid_svm.best_score_)

    print(report_results(grid_svm.best_estimator_, X_test, y_test))

else:
    svc = SVC(C=optimal_C, probability=True, kernel="linear", class_weight="balanced")
    tf_train = vect_count.fit_transform(X_train)
    tf_test = vect_count.transform(X_test)

    print("Count Vectorizer:")

    svc.fit(tf_train, y_train)
    svc.score(tf_train, y_train)

    y_pred = svc.predict(tf_test)

    print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
    print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred) * 100))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("Tfidf Vectorizer:")
    tf_train = vect_tfidf.fit_transform(X_train)
    tf_test = vect_tfidf.transform(X_test)

    svc.fit(tf_train, y_train)
    svc.score(tf_train, y_train)

    y_pred = svc.predict(tf_test)

    print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
    print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred) * 100))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
