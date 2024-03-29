# slangID3

slangID3 tries to detect slang phrases. Something literally no one asked for...

You can train a selection of classifiers, and print out a test set of phrases with the **DEMO** button.
Or you can pass a phrase and see what type it, and the individual words are identified as. All the models are pre-trained, but you can re-train if needed.

# What's new?

* New GUI with a modern look
* Integrated output window
* Data Augmentation to obtain larger data artificially (currently very limited)
* Individual word analysis with seperately trained models
* New data formatting
* New preprocessing

# Performance
In total, there are five classifiers you can choose from:

* Linear SVM (SVC with linear Kernel)
* Decision Tree
* Gaussian Naive Bayes
* Multinomial Naive Bayes
* Logistic Regression

Currently the **best performer** is the **Logistic Regression model** with an **F<sub>1</sub> score of ~96.10%**

(with augmented data of size 50. Might change with more diverse data. Currently biased towards "slang".)

# How to run slangID3

Download the .exe file under "Releases", **or**

1. Install Python **3.10** or later.
2. Install the required packages by running `pip install -r requirements.txt` in your shell of choice. Make sure you are in the project directory.
3. Run `python slangID3.py`

**Note:** It might take a while to load. Be patient.


# Gallery

______________________
**Icon**

<img src='https://raw.githubusercontent.com/m4cit/slangID3/main/misc/gallery/slangID3_icon.png' height="120">
______________________
_____________________________________________________________________________________________________________________________
**Prediction**

<img src='https://raw.githubusercontent.com/m4cit/slangID3/main/misc/gallery/slangID3_pred.png' width="900">
_____________________________________________________________________________________________________________________________

**Demo**

<img src='https://raw.githubusercontent.com/m4cit/slangID3/main/misc/gallery/slangID3_dm.png' width="900">


# Source of the data

Most of the phrases come from archive.org's [Twitter Stream of June 6th](https://archive.org/details/archiveteam-twitter-stream-2021-06).

# Recognition of Open Source use

* scikit-learn
* customtkinter
* pandas
* tqdm
* pyinstaller

