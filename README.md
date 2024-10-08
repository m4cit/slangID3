# slangID3
<img src='https://raw.githubusercontent.com/m4cit/slangID3/main/misc/gallery/slangID3_icon.png' align="left" height="150">
slangID3 tries to identify slang phrases.

You can train a selection of classifiers, and print out a test set of phrases with the **DEMO** button.
Or you can pass a phrase and see what type it, and the individual words are identified as. All the models are pre-trained, but you can re-train if needed.
<br clear="left">

## What's new?
* New GUI with a modern look
* Integrated output window
* Data Augmentation to obtain larger data artificially (currently very limited)
* Individual word evaluation
* New data formatting
* New preprocessing


## Gallery
### Prediction
<img src='https://raw.githubusercontent.com/m4cit/slangID3/main/misc/gallery/slangID3_pred.png' width="900">

### Demo
<img src='https://raw.githubusercontent.com/m4cit/slangID3/main/misc/gallery/slangID3_dm.png' width="900">


## How to run slangID3
1. Download the latest **slangID3.exe** and the source code files in [_releases_](https://github.com/m4cit/slangID3/releases).

2. Unzip the source code file.

3. Move **slangID3.exe** to the unzipped folder.

**or**
   
1. Install Python **3.10** or newer.

2. Install the required packages by running
   >```
   >pip install -r requirements.txt
   >```
   in your shell of choice. Make sure you are in the project directory.

3. Run
   >```
   >python slangID3.py
   >```

**Note:** It might take a while to load. Be patient.


## Usage
You can predict with the included pre-trained models, and re-train if needed.

Preprocessing is the last step before training a model.

If you want to use the original dataset **_data.csv_** or the augmented dataset **_augmented_data.csv_**, use the preprocessing function before training.


## Performance
In total, there are five models you can choose from (for now):

* Linear SVM (SVC with linear Kernel)
* Decision Tree
* Gaussian Naive Bayes
* Multinomial Naive Bayes
* Logistic Regression

Currently the **best performer** is the **Linear SVM** model with an **F<sub>1</sub> score of 71.4%**
(on the test set, with the original training data)


## Issues
The training dataset is still too small, resulting in overfitting (after augmentation).


## Augmentation
I categorized the slang words as:
* \<pex> personal expressions
  * _dude, one and only, bro_
* \<n> singular nouns
  * _shit_
* \<npl> plural nouns
  * _crybabies_
* \<shnpl> shortened plural nouns
  * _ppl_
* \<mwn> multiword nouns
  * _certified vaccine freak_
* \<mwexn> multiword nominal expressions
  * _a good one_
* \<en> exaggerated nouns
  * _guysssss_
* \<eex> (exaggerated) expressions
  * _hahaha, aaaaaah, lmao_
* \<adj> adjectives
  * _retarded_
* \<eadj> exaggerated adjectives
  * _weirdddddd_
* \<sha> shortened adjectives
  * _on_
* \<shmex> shortened (multiword) expressions
  * _tbh, imo_
* \<v> infinitive verb
  * _trigger_

(not all tags are available due to the small dataset)


## Preprocessing
The preprocessing script removes the slang tags, brackets, hyphens, and converts everything to lowercase.


## Source of the data
Most of the phrases come from archive.org's [Twitter Stream of June 6th](https://archive.org/details/archiveteam-twitter-stream-2021-06).


## Recognition of Open Source use
* scikit-learn
* customtkinter
* pandas
* tqdm
* pyinstaller

