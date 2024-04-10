from sklearn.naive_bayes import GaussianNB
from .modifiers.splitting import train_y, test_x, test_y, vect_train_x, vect_test_x
from sklearn.metrics import f1_score
import pickle
from tqdm import tqdm

# function to train the classifier
def dm_train_nbg():
    print('\nTraining Naive Bayes (Gaussian)...')
    message = 'Training Naive Bayes (Gaussian)...'
    progressbar = tqdm(total=3)
    clf_nbg = GaussianNB()
    progressbar.update()
    clf_nbg.fit(vect_train_x.toarray(), train_y)
    progressbar.update()
    # saving trained classifier
    with open('./classifiers/models/gaussian_nb_classifier.pkl', 'wb') as f:
        pickle.dump(clf_nbg,f)
    progressbar.update()
    progressbar.close()
    progressbar_str = str(progressbar)
    end = "\n"
    return message, progressbar_str, end

# function to predict the test set and list the phrases
def dm_naive_g():
    # loading pre-trained classifier
    try:
        with open('./classifiers/models/gaussian_nb_classifier.pkl', 'rb') as f:
            loaded_clf = pickle.load(f)
        print("\nNaive Bayes (Gaussian)\nTest Phrases: ")
        out1 = "Naive Bayes (Gaussian):"
        out2 = "|\nTest Phrases:\n"
        for i in range(len(test_x)):
            print("[" , i, "]", "pred:", loaded_clf.predict(vect_test_x[i].toarray()), "  ", test_x[i])
            out2 += "[" + str(i+1) + "]" + "pred:" + str(loaded_clf.predict(vect_test_x[i].toarray())) + "  " + test_x[i] + "\n"
        def f1():
            pred_y_nbg = loaded_clf.predict(vect_test_x.toarray())
            f1 = f1_score(test_y, pred_y_nbg,  average='macro', labels=['slang'])
            print("F1 score for slang: ", f1, "\n")
            out3 = "F1 score for slang: " + str(f1)
            return out3
        f1()
        end = "\n"
        return out1, f1(), out2, end
    except FileNotFoundError:
        print("Please train the model(s) first!")

