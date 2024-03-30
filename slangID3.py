import customtkinter as ctk
from classifiers.ignore_predict import predictor_svm, predictor_dec, predictor_naive_g, predictor_naive_m, predictor_log, check_svm, check_dec, check_log, check_naive_g, check_naive_m
from classifiers.ignore_predict import train_svm, train_dt, train_nbg, train_nbm, train_log
from classifiers.log_regression import dm_log_reg, dm_train_log
from classifiers.linear_svm import dm_lin_svm, dm_train_svm
from classifiers.decision_tree import dm_dec_tree, dm_train_dt
from classifiers.gaussian_nb import dm_naive_g, dm_train_nbg
from classifiers.multinomial_nb import dm_naive_m, dm_train_nbm
from classifiers.modifiers.data.augment import aug
from tkinter.filedialog import askopenfilename
from classifiers.modifiers.preprocessing import pp
from classifiers.modifiers.preprocessing import color
import os
import platform
from PIL import Image
import ctypes

# adjust scaling due to issues at 150%
scaleFactor=ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100
if scaleFactor == 1.25:
    ctk.set_window_scaling(0.95)
    ctk.set_widget_scaling(0.95)
if scaleFactor == 1.5:
    ctk.set_window_scaling(0.75)
    ctk.set_widget_scaling(0.75)
if scaleFactor == 1.75:
    ctk.set_window_scaling(0.55)
    ctk.set_widget_scaling(0.55)
if scaleFactor == 2.0:
    ctk.set_window_scaling(0.35)
    ctk.set_widget_scaling(0.35)

# training all models
def train_all(statSVM, statD, statG, statM, statLOG):
    if statSVM == False and statD == False and statG == False and statM == False and statLOG == False:
        print(color.BOLD,color.RED,"\nPlease choose a model to train!",color.END)
        info = '\nPlease choose a model to train!\n'
        output.insert("1.0", info)
    else:
        if check_log(statLOG) == True:
            out = train_log()
            output.insert("1.0", "\n".join(out))
        if check_naive_m(statM) == True:
            out = train_nbm()
            output.insert("1.0", "\n".join(out))
        if check_naive_g(statG) == True:
            out = train_nbg()
            output.insert("1.0", "\n".join(out))
        if check_dec(statD) == True:
            out = train_dt()
            output.insert("1.0", "\n".join(out))
        if check_svm(statSVM) == True:
            out = train_svm()
            output.insert("1.0", "\n".join(out))
        print(color.RED,color.BOLD,'\nPlease restart the application for changes to take effect!',color.END)
        info = '\nPlease restart the application for changes to take effect!\n\n'
        output.insert("1.0", info)

# running all models
def predictor_all(input, input2, statSVM, statD, statG, statM, statLOG):
    if statSVM == False and statD == False and statG == False and statM == False and statLOG == False:
        print(color.BOLD,color.RED,"\nPlease choose a model to use!",color.END)
        err = "\nPlease choose a model to use!\n"
        output.insert("1.0", err)
    if check_log(statLOG) == True:
        out = predictor_log(input, input2)
        output.insert("1.0", "\n".join(out))
    if check_naive_m(statM) == True:
        out = predictor_naive_m(input, input2)
        output.insert("1.0", "\n".join(out))
    if check_naive_g(statG) == True:
        out = predictor_naive_g(input, input2)
        output.insert("1.0", "\n".join(out))
    if check_dec(statD) == True:
        out = predictor_dec(input, input2)
        output.insert("1.0", "\n".join(out))
    if check_svm(statSVM) == True:
        out = predictor_svm(input, input2)
        output.insert("1.0", "\n".join(out))

        
# training all demo models
def dm_train_all():
    out =dm_train_log()
    output.insert("1.0", "\n".join(out))
    out =dm_train_nbm()
    output.insert("1.0", "\n".join(out))
    out = dm_train_nbg()
    output.insert("1.0", "\n".join(out))
    out = dm_train_dt()
    output.insert("1.0", "\n".join(out))
    out = dm_train_svm()
    output.insert("1.0", "\n".join(out))
    print(color.RED,color.BOLD,'\nPlease restart the application for changes to take effect!',color.END)
    info = '\nPlease restart the application for changes to take effect!\n\n'
    output.insert("1.0", info)

# running all demo models
def dm_predict_all():
    out = dm_log_reg()
    output.insert("1.0", "\n".join(out))
    out = dm_naive_m()
    output.insert("1.0", "\n".join(out))
    out = dm_naive_g()
    output.insert("1.0", "\n".join(out))
    out = dm_dec_tree()
    output.insert("1.0", "\n".join(out))
    out = dm_lin_svm()
    output.insert("1.0", "\n".join(out))
    

# open the window to select file
def openfilePP():
    try:
        file = askopenfilename(title='Select Training File', filetypes=[('Training File', '*.csv')], initialdir='./classifiers/modifiers/data/')
        out = pp(file)
        output.insert("1.0", "\n".join(out))
        # filename = os.path.basename(file)
        # # label for selected file
        # selectedTrainFile = tk.Label(root, text=filename, bg="white", font=("Calibri", 10), relief="ridge")
        # selectedTrainFile.place(width=220, height=40, x=350, y=331)
    except FileNotFoundError:
        print()
    
def openfileAugment(size):
    try:
        file = askopenfilename(title='Select Training File', filetypes=[('Training File', '*.csv')], initialdir='./classifiers/modifiers/data/')
        filename = os.path.basename(file)
        selectedfile = './classifiers/modifiers/data/'+filename
        # selectedTrainFileLabel = tk.Label(root, text=filename, bg="white", font=("Calibri", 10), relief="ridge")
        # selectedTrainFileLabel.place(width=220, height=40, x=20, y=331)
        out = aug(int(size), selectedfile)
        output.insert("1.0", "\n".join(out))
    except ValueError:
        print(color.BOLD,color.RED,'\nPlease enter an integer (size) first, then select the desired file!',color.END)
        err = 'Please enter an integer (size) first, then select the desired file!\n'
        output.insert("1.0", err)

    
# clear function for the shell / cmdlet
def clear():
    if platform.system() == "Windows":
        os.system("CLS")
    elif platform.system() == "Linux":
        os.system("clear")
    output.delete("1.0", ctk.END)


# initiating the  ctk module
root = ctk.CTk()
root.geometry("1600x800")
root.title("slangID3")
root.grid_columnconfigure((0), weight=1)
root.grid_columnconfigure((1), weight=1)

root.grid_rowconfigure((0), weight=1)
root.grid_rowconfigure((1), weight=1)
root.grid_rowconfigure((2), weight=1)
root.grid_rowconfigure((3), weight=1)
root.grid_rowconfigure((4), weight=1)


# output window
output = ctk.CTkTextbox(root, font=("Calibri", 20), width=780, fg_color="#3d3d3d", border_width=8, border_color="#8388d6", text_color="#e4e4e4", scrollbar_button_color="#e4e4e4")
output.grid(row=0, column=1, rowspan=5, columnspan=5, padx=15, pady=(10, 10), sticky="nwes")

checkbox_frame = ctk.CTkFrame(root, fg_color="transparent", border_color="#8388d6", border_width=2)
checkbox_frame.grid(row=1, rowspan=3, column=0, columnspan=2, padx=515, pady=(10, 0), sticky="nsw")

# field for text input
phrase_entry = ctk.CTkTextbox(root, font=("Calibri", 20), width=771, height=50, fg_color="#3d3d3d", border_width=8, border_color="#27b9ff", text_color="#e4e4e4", scrollbar_button_color="#e4e4e4")
phrase_entry.grid(row=0, column=0, padx=15, columnspan=2, pady=(10, 0), sticky="nsw")

# field for augmentation size input
aug_entry = ctk.CTkEntry(root, font=("Calibri", 20), width=80, height=80, fg_color="#3d3d3d", border_width=2, border_color="#e4e4e4", text_color="#e4e4e4", placeholder_text="size", placeholder_text_color="#e4e4e4")
aug_entry.grid(row=2, column=0, padx=160, pady=(0, 0), sticky="w")

icon = Image.open("./misc/gallery/slangID3_icon.png")
img = ctk.CTkImage(dark_image=icon, light_image=icon, size=(60,60))
img_label = ctk.CTkLabel(root, image=img, text="")
img_label.grid(row=4, column=0, padx=15, pady=(0, 10), sticky="sw")

# version number
version = ctk.CTkLabel(root, text="1.0", fg_color="transparent", text_color="white", font=("Calibri", 20))
version.grid(row=4, column=0, padx=80, pady=(0, 5), sticky="sw")


# Buttons and checkboxes
################################################################################################################################
use_button = ctk.CTkButton(root, command=lambda: predictor_all([phrase_entry.get("1.0",ctk.END)],phrase_entry.get("1.0",ctk.END),svm_checkbox.get(),dt_checkbox.get(),nbg_checkbox.get(),nbm_checkbox.get(),log_checkbox.get()), text="Use", font=("Calibri", 22,"bold"), text_color="black", fg_color="#27b9ff", hover_color="#b6c2fe", border_width=2, border_color="#e4e4e4", border_spacing=10)
use_button.grid(row=1, rowspan=1, column=0, padx=15, pady=10, sticky="nsw")

train_button = ctk.CTkButton(root, command=lambda: ([train_all(svm_checkbox.get(), dt_checkbox.get(), nbg_checkbox.get(), nbm_checkbox.get(), log_checkbox.get())]), text="Train", font=("Calibri", 22,"bold"), text_color="black", fg_color="#27b9ff", hover_color="#b6c2fe", border_width=2, border_color="#e4e4e4", border_spacing=10)
train_button.grid(row=1, rowspan=1, column=0, padx=170, pady=10, sticky="nsw")

augment_button = ctk.CTkButton(root, command=lambda: openfileAugment(aug_entry.get()), text="Augment\nData", font=("Calibri", 20,"bold"), text_color="white", fg_color="#8388d6", hover_color="#95a8fe", border_width=2, border_spacing=10, border_color="#e4e4e4")
augment_button.grid(row=2, column=0, rowspan=1, padx=15, pady=10, sticky="nsw")

preprocess_button = ctk.CTkButton(root, command=lambda: openfilePP(), text="Preprocess\nData", font=("Calibri", 20,"bold"), text_color="white", fg_color="#8388d6", hover_color="#95a8fe", border_width=2, border_spacing=10, border_color="#e4e4e4")
preprocess_button.grid(row=3, column=0, rowspan=1, padx=15, pady=10, sticky="nsw")

demo_train_button = ctk.CTkButton(root, command=lambda: dm_train_all(), text="Train\nDEMO", font=("Calibri", 20,"bold"), text_color="white", fg_color="#8388d6", hover_color="#95a8fe", border_width=2, border_spacing=10, border_color="#e4e4e4")
demo_train_button.grid(row=3, column=0, rowspan=1, padx=170, pady=10, sticky="nsw")

demo_button = ctk.CTkButton(root, command=lambda: dm_predict_all(), text="DEMO", font=("Calibri", 20,"bold"), text_color="white", fg_color="#8388d6", hover_color="#95a8fe", border_width=2, border_spacing=10, border_color="#e4e4e4")
demo_button.grid(row=3, column=0, rowspan=1, padx=325, pady=10, sticky="nsw")

clear_button = ctk.CTkButton(root, command=lambda: clear(), text="Clear Output", font=("Calibri", 20,"bold"), text_color="white", fg_color="#bf0606", hover_color="#930b0b", border_width=2, border_spacing=10, border_color="#e4e4e4")
clear_button.grid(row=1, column=0, rowspan=1, padx=325, pady=10, sticky="nsw")


svm_checkbox = ctk.CTkCheckBox(checkbox_frame, command=lambda: check_svm(svm_checkbox.get()), text="Linear SVM", font=("Calibri", 20,"bold"), text_color="white", corner_radius=10, fg_color="#b6c2fe", hover_color="#99c8fc", border_color="white")
svm_checkbox.grid(row=1, column=0, padx=10, pady=(10,0), sticky="nsw")

dt_checkbox = ctk.CTkCheckBox(checkbox_frame, command=lambda: check_dec(dt_checkbox.get()), text="Decision Tree", font=("Calibri", 20,"bold"), text_color="white", corner_radius=10, fg_color="#b6c2fe", hover_color="#99c8fc", border_color="white")
dt_checkbox.grid(row=2, column=0, padx=10, pady=(15,0), sticky="nsw")

nbg_checkbox = ctk.CTkCheckBox(checkbox_frame, command=lambda: check_naive_g(nbg_checkbox.get()), text="Naive Bayes (Gaussian)", font=("Calibri", 20,"bold"), text_color="white", corner_radius=10, fg_color="#b6c2fe", hover_color="#99c8fc", border_color="white")
nbg_checkbox.grid(row=3, column=0, padx=10, pady=(15,0), sticky="nsw")

nbm_checkbox = ctk.CTkCheckBox(checkbox_frame, command=lambda: check_naive_m(nbm_checkbox.get()), text="Naive Bayes (Multinomial)", font=("Calibri", 20,"bold"), text_color="white", corner_radius=10, fg_color="#b6c2fe", hover_color="#99c8fc", border_color="white")
nbm_checkbox.grid(row=4, column=0, padx=10, pady=(15,0), sticky="nsw")

log_checkbox = ctk.CTkCheckBox(checkbox_frame, command=lambda: check_log(log_checkbox.get()), text="Logistic Regression", font=("Calibri", 20,"bold"), text_color="white", corner_radius=10, fg_color="#b6c2fe", hover_color="#99c8fc", border_color="white")
log_checkbox.grid(row=5, column=0, padx=10, pady=(15,0), sticky="nsw")

select_all_checkbox = ctk.CTkCheckBox(checkbox_frame, command=lambda: (svm_checkbox.toggle(),dt_checkbox.toggle(),nbg_checkbox.toggle(),nbm_checkbox.toggle(),log_checkbox.toggle()), text="All models", font=("Calibri", 20,"bold"), text_color="white", border_color="red", corner_radius=10, fg_color="#b6c2fe", hover_color="#99c8fc")
select_all_checkbox.grid(row=6, column=0, padx=10, pady=(15,10), sticky="nsw")
################################################################################################################################

root.mainloop()

