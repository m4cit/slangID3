# slangID3
slangID3 tries to detect slang phrases. Something literally no one asked for...

You can train a selection of classifiers, and print out a test set of phrases with the **DEMO** button.
Or you can pass a phrase and see what type it, and the individual words are identified as. All the models are pre-trained, but you can re-train if needed.


## What's new?
* New GUI with a modern look
* Integrated output window
* Data Augmentation to obtain larger data artificially (currently very limited)
* Individual word analysis with seperately trained models
* New data formatting
* New preprocessing


## Augmentation
I categorized the slang words as:
* \<pex> personal expressions
* \<n> singular nouns
* \<npl> plural nouns
* \<shnpl> shortened plural nouns
* \<mwn> multiword nouns
* \<mwexn> multiword nominal expressions
* \<en> exaggerated nouns
* \<eex> (exaggerated) expressions
* \<adj> adjectives
* \<eadj> exaggerated adjectives
* \<sha> shortened adjectives
* \<shmex> shortened (multiword) expressions
* \<v> infinitive verb

(not all tags are available due to the small dataset)


## Preprocessing
The preprocessing script removes the slang tags, brackets, hyphens, and converts everything to lowercase.


## Issues
The training dataset is still too small, resulting in overfitting (after augmentation).


## Usage
Preprocessing is the last step before training a model. 

If you want to use the original dataset **_data.csv_** or the augmented dataset **_augmented_data.csv_**, use the preprocessing function before training.


## Performance
In total, there are five models you can choose from (for now):

* Linear SVM (SVC with linear Kernel)
* Decision Tree
* Gaussian Naive Bayes
* Multinomial Naive Bayes
* Logistic Regression

Currently the **best performers** are the **Decision Tree** and **Naive Bayes (Gaussian)** models with an **F<sub>1</sub> score of 64%**
(on the test set, with the original training data)


## How to run slangID3
1. Download the **slangID3.exe** and the **.zip** file under "Releases"
2. Unzip the **.zip** file
3. Move **slangID3.exe** to the unzipped folder

**or**

1. Install Python **3.10** or later.
2. Install the required packages by running `pip install -r requirements.txt` in your shell of choice. Make sure you are in the project directory.
3. Run `python slangID3.py`

**Note:** It might take a while to load. Be patient.


## Gallery

### Icon

<img src='https://raw.githubusercontent.com/m4cit/slangID3/main/misc/gallery/slangID3_icon.png' height="120">


### Prediction

<img src='https://raw.githubusercontent.com/m4cit/slangID3/main/misc/gallery/slangID3_pred.png' width="900">


### Demo

<img src='https://raw.githubusercontent.com/m4cit/slangID3/main/misc/gallery/slangID3_dm.png' width="900">


## Source of the data
Most of the phrases come from archive.org's [Twitter Stream of June 6th](https://archive.org/details/archiveteam-twitter-stream-2021-06).


## Recognition of Open Source use
* scikit-learn
* customtkinter
* pandas
* tqdm
* pyinstaller

