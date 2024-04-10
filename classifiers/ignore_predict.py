import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB   
from .modifiers.preprocessing import color
import pickle
from tqdm import tqdm

################ splitting the data #########################################################

data = pd.read_csv (r'./classifiers/modifiers/data/processed_data.csv')
train, test = train_test_split(data, test_size=0.1, random_state=42)
train_x = [x for x in train['phrase']]
train_y = [x for x in train['type']]
test_x = [x for x in test['phrase']]
test_y = [x for x in test['type']]

vectorizer = TfidfVectorizer()
vect_train_x = vectorizer.fit_transform(train_x)
vect_test_x = vectorizer.transform(test_x)

############## functions to train classifiers ################

def train_svm():
    print('\nTraining Linear SVM...')
    message = 'Training Linear SVM...'
    progressbar = tqdm(total=3)
    clf_svm = svm.SVC(kernel='linear')
    clf_svm_tk = svm.SVC(kernel='linear')
    progressbar.update()
    clf_svm.fit(vect_train_x, train_y)
    clf_svm_tk.fit(vect_train_x, train_y)
    progressbar.update()
    # saving trained models
    with open('./classifiers/models/pd_linear_svm_classifier.pkl', 'wb') as f:
        with open('./classifiers/models/pd_linear_svm_classifier_tk.pkl', 'wb') as f2:
            pickle.dump(clf_svm,f)
            pickle.dump(clf_svm_tk,f2)
    progressbar.update()
    progressbar.close()
    progressbar_str = str(progressbar)
    end = "\n"
    return message, progressbar_str, end

def train_log():
    print('\nTraining Logistic Regression...')
    message = 'Training Logistic Regression...'
    progressbar = tqdm(total=3)
    clf_log = LogisticRegression(C=32, fit_intercept=False, solver='newton-cg')
    clf_log_tk = LogisticRegression(C=32, fit_intercept=False, solver='newton-cg')
    progressbar.update()
    clf_log.fit(vect_train_x, train_y)
    clf_log_tk.fit(vect_train_x, train_y)
    progressbar.update()
    # saving trained models
    with open('./classifiers/models/pd_log_regression_classifier.pkl', 'wb') as f:
        with open('./classifiers/models/pd_log_regression_classifier_tk.pkl', 'wb') as f2:
            pickle.dump(clf_log,f)
            pickle.dump(clf_log_tk,f2)
    progressbar.update()
    progressbar.close()
    progressbar_str = str(progressbar)
    end = "\n"
    return message, progressbar_str, end

def train_dt():
    print('\nTraining Decision Tree...')
    message = 'Training Decision Tree...'
    progressbar = tqdm(total=3)
    clf_dt = DecisionTreeClassifier(min_samples_split=2)
    clf_dt_tk = DecisionTreeClassifier(min_samples_split=2)
    progressbar.update()
    clf_dt.fit(vect_train_x, train_y)
    clf_dt_tk.fit(vect_train_x, train_y)
    progressbar.update()
    # saving trained models
    with open('./classifiers/models/pd_decision_tree_classifier.pkl', 'wb') as f:
        with open('./classifiers/models/pd_decision_tree_classifier_tk.pkl', 'wb') as f2:
            pickle.dump(clf_dt,f)
            pickle.dump(clf_dt_tk,f2)
    progressbar.update()
    progressbar.close()
    progressbar_str = str(progressbar)
    end = "\n"
    return message, progressbar_str, end

def train_nbg():
    print('\nTraining Naive Bayes (Gaussian)...')
    message = 'Training Naive Bayes (Gaussian)...'
    progressbar = tqdm(total=3)
    clf_nbg = GaussianNB()
    clf_nbg_tk = GaussianNB()
    progressbar.update()
    clf_nbg.fit(vect_train_x.toarray(), train_y)
    clf_nbg_tk.fit(vect_train_x.toarray(), train_y)
    progressbar.update()
    # saving trained models
    with open('./classifiers/models/pd_gaussian_nb_classifier.pkl', 'wb') as f:
        with open('./classifiers/models/pd_gaussian_nb_classifier_tk.pkl', 'wb') as f2:
            pickle.dump(clf_nbg,f)
            pickle.dump(clf_nbg_tk,f2)
    progressbar.update()
    progressbar.close()
    progressbar_str = str(progressbar)
    end = "\n"
    return message, progressbar_str, end

def train_nbm():
    print('\nTraining Naive Bayes (Multinomial)...')
    message = 'Training Naive Bayes (Multinomial)...'
    progressbar = tqdm(total=3)
    clf_nbm = MultinomialNB(alpha=2.0, fit_prior=False)
    clf_nbm_tk = MultinomialNB(alpha=2.0, fit_prior=False)
    progressbar.update()
    clf_nbm.fit(vect_train_x, train_y)
    clf_nbm_tk.fit(vect_train_x, train_y)
    progressbar.update()
    # saving trained models
    with open('./classifiers/models/pd_multinomial_nb_classifier.pkl', 'wb') as f:
        with open('./classifiers/models/pd_multinomial_nb_classifier_tk.pkl', 'wb') as f2:
            pickle.dump(clf_nbm,f)
            pickle.dump(clf_nbm_tk,f2)
    progressbar.update()
    progressbar.close()
    progressbar_str = str(progressbar)
    end = "\n"
    return message, progressbar_str, end

####################################################################################

# checkbox functions
def check_svm(stat):
    if stat == 0:
        return False
    else:
        return True
    
def check_log(stat):
    if stat == 0:
        return False
    else:
        return True

def check_dec(stat):
    if stat == 0:
        return False
    else:
        return True
    
def check_naive_g(stat):
    if stat == 0:
        return False
    else:
        return True
    
def check_naive_m(stat):
    if stat == 0:
        return False
    else:
        return True
    
############# following functions to predict the input phrase, as well as print the mean accuracy of the classifier##############
def predictor_svm(input, input2):
    try:
        # loading pre-trained classifier
        with open('./classifiers/models/pd_linear_svm_classifier.pkl','rb') as f:
            loaded_clf = pickle.load(f)
            print(color.BOLD,"\nLinear SVM:",color.END,"\n Phrase:\t " + input[0], "Prediction:\t", loaded_clf.predict(vectorizer.transform(input))[0])
            print(" mean accuracy:\t", loaded_clf.score(vect_test_x, test_y))
            out1 = "Linear SVM: \n  Phrase:\t         " + input[0] + "  |--Prediction: " + loaded_clf.predict(vectorizer.transform(input))[0] + "\n  |"
            
        # loading pre-trained classifier for tokens only
        with open('./classifiers/models/pd_linear_svm_classifier_tk.pkl','rb') as f2:
            loaded_clf_tk = pickle.load(f2)
            words = input2.split(' ')
            out2 = ""
            for w in range(len(words)):
                index = "  [" + str(w+1) + "]"
                print("\n\t\tWord:\t\t", words[w].replace('\n', ''), "\n\t\tPrediction:\t", loaded_clf_tk.predict(vectorizer.transform(words)[w])[0])
                print("\t\tmean accuracy:\t", loaded_clf_tk.score(vect_test_x, test_y))
                out2 += index + " Word:\t   " + str(words[w].replace('\n', '')) + "\n  Prediction:\t " + str(loaded_clf_tk.predict(vectorizer.transform(words)[w])[0]) + "\n"
            end = "\n"
        return out1, out2, end
    except FileNotFoundError:
        print("Please train the model(s) first!")


def predictor_log(input, input2):
    try:
        # loading pre-trained classifier
        with open('./classifiers/models/pd_log_regression_classifier.pkl','rb') as f:
            loaded_clf = pickle.load(f)
            print(color.BOLD,"\nLogistic Regression:",color.END,"\n Phrase:\t " + input[0], "Prediction:\t", loaded_clf.predict(vectorizer.transform(input))[0])
            print(" mean accuracy:\t", loaded_clf.score(vect_test_x, test_y))
            out1 = "Logistic Regression: \n  Phrase:\t         " + input[0] + "  |--Prediction: " + loaded_clf.predict(vectorizer.transform(input))[0] + "\n  |"
            
        # loading pre-trained classifier for tokens only
        with open('./classifiers/models/pd_log_regression_classifier_tk.pkl','rb') as f2:
            loaded_clf_tk = pickle.load(f2)
            words = input2.split(' ')
            out2 = ""
            for w in range(len(words)):
                index = "  [" + str(w+1) + "]"
                print("\n\t\tWord:\t\t", words[w].replace('\n', ''), "\n\t\tPrediction:\t", loaded_clf_tk.predict(vectorizer.transform(words)[w])[0])
                print("\t\tmean accuracy:\t", loaded_clf_tk.score(vect_test_x, test_y))
                out2 += index + " Word:\t   " + str(words[w].replace('\n', '')) + "\n  Prediction:\t " + str(loaded_clf_tk.predict(vectorizer.transform(words)[w])[0]) + "\n"
            end = "\n"
            return out1, out2, end
    except FileNotFoundError:
        print("Please train the model(s) first!")


def predictor_dec(input, input2):
    try:
        # loading pre-trained classifier
        with open('./classifiers/models/pd_decision_tree_classifier.pkl','rb') as f:
            loaded_clf = pickle.load(f)
            print(color.BOLD,"\nDecision Tree:",color.END,"\n Phrase:\t " + input[0], "Prediction:\t", loaded_clf.predict(vectorizer.transform(input))[0])
            print(" mean accuracy:\t", loaded_clf.score(vect_test_x, test_y))
            out1 = "Decision Tree: \n  Phrase:\t         " + input[0] + "  |--Prediction: " + loaded_clf.predict(vectorizer.transform(input))[0] + "\n  |"
            
    # loading pre-trained classifier for tokens only
        with open('./classifiers/models/pd_decision_tree_classifier_tk.pkl','rb') as f2:
            loaded_clf_tk = pickle.load(f2)
            words = input2.split(' ')
            out2 = ""
            for w in range(len(words)):
                index = "  [" + str(w+1) + "]"
                print("\n\t\tWord:\t\t", words[w].replace('\n', ''), "\n\t\tPrediction:\t", loaded_clf_tk.predict(vectorizer.transform(words)[w])[0])
                print("\t\tmean accuracy:\t", loaded_clf_tk.score(vect_test_x, test_y))
                out2 += index + " Word:\t   " + str(words[w].replace('\n', '')) + "\n  Prediction:\t " + str(loaded_clf_tk.predict(vectorizer.transform(words)[w])[0]) + "\n"
            end = "\n"
            return out1, out2, end
    except FileNotFoundError:
        print("Please train the model(s) first!")
        

def predictor_naive_g(input, input2):
    try:
        # loading pre-trained classifier
        with open('./classifiers/models/pd_gaussian_nb_classifier.pkl','rb') as f:
            loaded_clf = pickle.load(f)
            print(color.BOLD,"\nNaive Bayes (Gaussian):",color.END,"\n Phrase:\t " + input[0], "Prediction:\t", loaded_clf.predict(vectorizer.transform(input).toarray())[0])
            print(" mean accuracy:\t", loaded_clf.score(vect_test_x.toarray(), test_y))
            out1 = "Naive Bayes (Gaussian): \n  Phrase:\t         " + input[0] + "  |--Prediction: " + loaded_clf.predict(vectorizer.transform(input).toarray())[0] + "\n  |"
        
        # loading pre-trained classifier for tokens only
        with open('./classifiers/models/pd_gaussian_nb_classifier_tk.pkl','rb') as f2:
            loaded_clf_tk = pickle.load(f2)
            words = input2.split(' ')
            out2 = ""
            for w in range(len(words)):
                index = "  [" + str(w+1) + "]"
                print("\n\t\tWord:\t\t", words[w].replace('\n', ''), "\n\t\tPrediction:\t", loaded_clf_tk.predict(vectorizer.transform(words)[w].toarray())[0])
                print("\t\tmean accuracy:\t", loaded_clf_tk.score(vect_test_x.toarray(), test_y))
                out2 += index + " Word:\t   " + str(words[w].replace('\n', '')) + "\n  Prediction:\t " + str(loaded_clf_tk.predict(vectorizer.transform(words)[w].toarray())[0]) + "\n"
            end = "\n"
            return out1, out2, end
    except FileNotFoundError:
        print("Please train the model(s) first!")
        

def predictor_naive_m(input, input2):
    try:
        # loading pre-trained classifier
        with open('./classifiers/models/pd_multinomial_nb_classifier.pkl','rb') as f:
            loaded_clf = pickle.load(f)
            print(color.BOLD,"\nNaive Bayes (Multinomial):",color.END,"\n Phrase:\t " + input[0], "Prediction:\t", loaded_clf.predict(vectorizer.transform(input))[0])
            print(" mean accuracy:\t", loaded_clf.score(vect_test_x, test_y))
            out1 = "Naive Bayes (Multinomial): \n  Phrase:\t         " + input[0] + "  |--Prediction: " + loaded_clf.predict(vectorizer.transform(input))[0] + "\n  |"
    
    # loading pre-trained classifier for tokens only
        with open('./classifiers/models/pd_multinomial_nb_classifier_tk.pkl','rb') as f2:
            loaded_clf_tk = pickle.load(f2)
            words = input2.split(' ')
            out2 = ""
            for w in range(len(words)):
                index = "  [" + str(w+1) + "]"
                print("\n\t\tWord:\t\t", words[w].replace('\n', ''), "\n\t\tPrediction:\t", loaded_clf_tk.predict(vectorizer.transform(words)[w])[0])
                print("\t\tmean accuracy:\t", loaded_clf_tk.score(vect_test_x, test_y))
                out2 += index + " Word:\t   " + str(words[w].replace('\n', '')) + "\n  Prediction:\t " + str(loaded_clf_tk.predict(vectorizer.transform(words)[w])[0]) + "\n"
            end = "\n"
            return out1, out2, end
    except FileNotFoundError:
        print("Please train the model(s) first!")
        
