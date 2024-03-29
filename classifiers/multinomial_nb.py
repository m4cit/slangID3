from sklearn.naive_bayes import MultinomialNB
from .modifiers.splitting import train_y, test_x, test_y, vect_train_x, vect_test_x
from sklearn.metrics import f1_score
import pickle
from tqdm import tqdm

# function to train the classifier
def dm_train_nbm():
    print('\nTraining Naive Bayes (Multinomial)...')
    message = 'Training Naive Bayes (Multinomial)...'
    progressbar = tqdm(total=3)
    clf_nbm = MultinomialNB(alpha=2.0, fit_prior=False)
    progressbar.update()
    clf_nbm.fit(vect_train_x, train_y)
    progressbar.update()
    # saving trained classifier
    with open('./classifiers/models/multinomial_nb_classifier.pkl', 'wb') as f:
        pickle.dump(clf_nbm,f)
    progressbar.update()
    progressbar.close()
    progressbar_str = str(progressbar)
    end = "\n"
    return message, progressbar_str, end

# function to predict the test set and list the phrases
def dm_naive_m():
    try:
        # loading pre-trained classifier
        with open('./classifiers/models/multinomial_nb_classifier.pkl', 'rb') as f:
            loaded_clf = pickle.load(f)
        print("\nNaive Bayes (Multinomial)\nTest Phrases: ")
        out1 = "Naive Bayes (Multinomial):"
        out2 = "|\nTest Phrases:\n"
        for i in range(len(test_x)):
            print("[" , i, "]", "pred:", loaded_clf.predict(vect_test_x[i]), "  ", test_x[i])
            out2 += "[" + str(i+1) + "]" + "pred:" + str(loaded_clf.predict(vect_test_x[i])) + "  " + test_x[i] + "\n"
        def test_score():
            print("\nmean accuracy:      ", loaded_clf.score(vect_test_x, test_y))
            out3 = "mean accuracy: " + str(loaded_clf.score(vect_test_x, test_y))
            return out3
        def f1():
            pred_y_nbm = loaded_clf.predict(vect_test_x)
            f1 = f1_score(test_y, pred_y_nbm,  average='macro', labels=['slang'])
            print("F1 score for slang: ", f1, "\n")
            out4 = "F1 score for slang: " + str(f1)
            return out4
        test_score()
        f1()
        end = "\n"
        return out1, test_score(), f1(), out2, end
    except FileNotFoundError:
        print("Please train the model(s) first!")

