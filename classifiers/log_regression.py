from sklearn.linear_model import LogisticRegression
from .modifiers.splitting import train_y, test_x, test_y, vect_train_x, vect_test_x
from sklearn.metrics import f1_score
import pickle
from tqdm import tqdm

# function to train the classifier
def dm_train_log():
    print('\nTraining Logistic Regression...')
    message = 'Training Logistic Regression...'
    progressbar = tqdm(total=3)
    clf_log = LogisticRegression(C=32, fit_intercept=False, solver='newton-cg')
    progressbar.update()
    clf_log.fit(vect_train_x, train_y)
    progressbar.update()
    # saving trained classifier
    with open('./classifiers/models/log_regression_classifier.pkl', 'wb') as f:
        pickle.dump(clf_log,f)
    progressbar.update()
    progressbar.close()
    progressbar_str = str(progressbar)
    end = "\n"
    return message, progressbar_str, end

# function to predict the test set and list the phrases, as well as the predictions
def dm_log_reg():
    # loading pre-trained classifier
    try:
        with open('./classifiers/models/log_regression_classifier.pkl', 'rb') as f:
            loaded_clf = pickle.load(f)
        print("\nLogistic Regression\nTest Phrases: ")
        out1 = "Logistic Regression:"
        out2 = "|\nTest Phrases:\n"
        for i in range(len(test_x)):
            print("[" , i, "]", "pred:", loaded_clf.predict(vect_test_x[i]), "  ", test_x[i])
            out2 += "[" + str(i+1) + "]" + "pred:" + str(loaded_clf.predict(vect_test_x[i])) + "  " + test_x[i] + "\n"
        def test_score():
            print("\nmean accuracy:      ", loaded_clf.score(vect_test_x, test_y))
            out3 = "mean accuracy: " + str(loaded_clf.score(vect_test_x, test_y))
            return out3
        def f1():
            pred_y_log = loaded_clf.predict(vect_test_x)
            f1 = f1_score(test_y, pred_y_log,  average='macro', labels=['slang'])
            print("F1 score for slang: ", f1, "\n")
            out4 = "F1 score for slang: " + str(f1)
            return out4
        test_score()
        f1()
        end = "\n"
        return out1, test_score(), f1(), out2, end
    except FileNotFoundError:
        print("Please train the model(s) first!")

