from sklearn.tree import DecisionTreeClassifier
from .modifiers.splitting import train_y, test_x, test_y, vect_train_x, vect_test_x
from sklearn.metrics import f1_score
import pickle
from tqdm import tqdm

# function to train the classifier
def dm_train_dt():
    print('\nTraining Decision Tree...')
    message = 'Training Decision Tree...'
    progressbar = tqdm(total=3)
    clf_dt = DecisionTreeClassifier(min_samples_split=2)
    progressbar.update()
    clf_dt.fit(vect_train_x, train_y)
    progressbar.update()
    # saving trained classifier
    with open('./classifiers/models/decision_tree_classifier.pkl', 'wb') as f:
        pickle.dump(clf_dt,f)
    progressbar.update()
    progressbar.close()
    progressbar_str = str(progressbar)
    end = "\n"
    return message, progressbar_str, end

# function to predict the test set and list the phrases
def dm_dec_tree():
    # loading pre-trained classifier
    try:
        with open('./classifiers/models/decision_tree_classifier.pkl', 'rb') as f:
            loaded_clf = pickle.load(f)
        print("\nDecision Tree\nTest Phrases: ")
        out1 = "Decision Tree:"
        out2 = "|\nTest Phrases:\n"
        for i in range(len(test_x)):
            print("[" , i, "]", "pred:", loaded_clf.predict(vect_test_x[i]), "  ", test_x[i])
            out2 += "[" + str(i+1) + "]" + "pred:" + str(loaded_clf.predict(vect_test_x[i])) + "  " + test_x[i] + "\n"
        def test_score():
            print("\nmean accuracy:      ", loaded_clf.score(vect_test_x, test_y))
            out3 = "mean accuracy: " + str(loaded_clf.score(vect_test_x, test_y))
            return out3
        def f1():
            pred_y_dt = loaded_clf.predict(vect_test_x)
            f1 = f1_score(test_y, pred_y_dt,  average='macro', labels=['slang'])
            print("F1 score for slang: ", f1, "\n")
            out4 = "F1 score for slang: " + str(f1)
            return out4
        test_score()
        f1()
        end = "\n"
        return out1, test_score(), f1(), out2, end
    except FileNotFoundError:
        print("Please train the model(s) first!")

