#!/usr/bin/env python
# coding: utf-8

# # Speech Recognition Assignment

# ## Data Pre-processing

# ### Package Importing

# In[4]:


import wave, os, glob, csv, sys, pathlib, math, gc
from random import shuffle
import pandas as pd
import numpy as np
import pickle
from numpy import save, load, savez
from python_speech_features import mfcc

import scipy.stats as stats
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile

from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Visualization
import IPython.display as ipd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# import seaborn as sns

import h5py
# import cv2

get_ipython().run_line_magic('matplotlib', 'inline')
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# ## Let's have a look at the files in our dataset

# In[5]:


#Obtain list of dataset, labels and speakers
this_directory = os.getcwd()
filelist = []
filepathlist = []
ids = []

def get_file_paths(dirname):
    file_paths = []
    subdir_paths = []
    for root, directories, files in os.walk(dirname):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    return file_paths    

def get_dataset_list():
    files = get_file_paths(this_directory)                 
    for file in files:                              
        (filepath, ext) = os.path.splitext(file)    
        file_name = os.path.basename(file)
        sub_directory = os.path.dirname(file)
        sub_directory = sub_directory.split('/')[-1]
        if ext == '.wav':                           
            filelist.append(file_name)
            filepathlist.append(sub_directory)
    for file in filelist:
        ids.append(file[0:7])
    dataset_list = list(zip(filelist, filepathlist, ids))
    return dataset_list

if __name__ == '__main__':
    dataset_list = get_dataset_list()


# In[6]:


#Looking at training + test + validation sizes
training_size = round(0.7*len(dataset_list))
validation_size = round(0.1*len(dataset_list))
test_size = len(dataset_list) - (training_size + validation_size)

#Sanity Check
print(training_size + validation_size + test_size)
print(len(dataset_list))

#Convert to dataframe, create labels list
dataset_df = pd.DataFrame(dataset_list, columns = ("FileName", "Label", "Speaker"))
labels = set(filepathlist)
print(labels)
print(len(labels))


# In[7]:


# Plot Dataset for Imbalance Check
f, ax = plt.subplots(figsize=(15,6))
sns.set_style("whitegrid")
sns.countplot(x="Label",
data=dataset_df)


# Looking at unusable recordings shorter than 1 second and removing them

# In[8]:


duration_of_recordings=[]
unusable_files = []
waves = [file for file in dataset_df["FileName"]]
for index, row in dataset_df.iterrows():
    sample_rate, samples = wavfile.read(os.getcwd()+ "/" + row['Label'] + "/" + row['FileName'])
    duration_of_recordings.append(float(len(samples)/sample_rate))
    if samples.shape[0] < sample_rate:
        unusable_files.append(row['Label'] + "/" + row['FileName'])
        dataset_df.drop(index, inplace=True)
    gc.collect

plt.hist(np.array(duration_of_recordings), range=(0.6,1.3))
plt.xlabel('Duration in seconds')
plt.ylabel('Quantity of files')
plt.savefig('duration.png', bbox_inches='tight')
plt.show()


# In[9]:


# Plot Dataset for Imbalance Check after dropping unusable files
f, ax = plt.subplots(figsize=(15,6))
sns.set_style("whitegrid")
imb = sns.countplot(x="Label",
                    data=dataset_df,
                    palette="Greens_d")
plt.savefig('dataimb.png', bbox_inches='tight')


# In[10]:


#Split into separate dataframes
grouped_df = dataset_df.groupby(dataset_df.Label)

house_df = grouped_df.get_group("house")
cat_df = grouped_df.get_group("cat")
bird_df = grouped_df.get_group("bird")
left_df = grouped_df.get_group("left")
off_df = grouped_df.get_group("off")
dog_df = grouped_df.get_group("dog")
marvin_df = grouped_df.get_group("marvin")
backward_df = grouped_df.get_group("backward")
go_df = grouped_df.get_group("go")
visual_df = grouped_df.get_group("visual")

df_sublist = []
df_train_sublist = []
df_testval_sublist = []
for label in labels:
    df_sublist.append(("{}_df").format(label))
    df_train_sublist.append(("{}_df_train").format(label))
    df_testval_sublist.append(("{}_df_testval").format(label))


# In[11]:


print(df_sublist)
print(df_train_sublist)


# In[12]:


def check_duplicates_to_training_set(df):
    if len(df[df.duplicated("Speaker", keep=False)]) <= (0.7* len(df)):
        print("{:.2%}".format(len(df[df.duplicated("Speaker", keep=False)])/len(df)))
    else: 
        print("{:.2%}".format(len(df[df.duplicated("Speaker", keep=False)])/len(df)))
    return


# In[13]:


#Choosing files for training set and checking the 70% split for data balance
house_df_train =check_duplicates_to_training_set(house_df)
cat_df_train =check_duplicates_to_training_set(cat_df)
bird_df_train =check_duplicates_to_training_set(bird_df)
left_df_train =check_duplicates_to_training_set(left_df)
off_df_train =check_duplicates_to_training_set(off_df)
dog_df_train =check_duplicates_to_training_set(dog_df)
marvin_df_train =check_duplicates_to_training_set(marvin_df)
backward_df_train = check_duplicates_to_training_set(backward_df)
go_df_train =check_duplicates_to_training_set(go_df)
visual_df_train =check_duplicates_to_training_set(visual_df)


# In[14]:


#How many recordings have others by the same speaker
num_of_dupes = 0
for df in df_sublist:
    print(len(eval(df)[eval(df).duplicated("Speaker", keep=False)]))
    num_of_dupes += (len(eval(df)[eval(df).duplicated("Speaker", keep=False)]))


# In[15]:


print("{:.2%}".format(num_of_dupes/len(dataset_list)))
#Since this is below 70% we could proceed to add all the data to training


# In[16]:


def add_duplicates_to_training_set(df):
    return df[df.duplicated("Speaker", keep=False)], df.drop_duplicates(subset="Speaker", keep=False)


# In[17]:


#Populating training and test/val dataframes
house_df_train, house_df_testval = add_duplicates_to_training_set(house_df)
cat_df_train,cat_df_testval = add_duplicates_to_training_set(cat_df)
bird_df_train, bird_df_testval = add_duplicates_to_training_set(bird_df)
left_df_train, left_df_testval = add_duplicates_to_training_set(left_df)
off_df_train, off_df_testval = add_duplicates_to_training_set(off_df)
dog_df_train, dog_df_testval = add_duplicates_to_training_set(dog_df)
marvin_df_train, marvin_df_testval = add_duplicates_to_training_set(marvin_df)
backward_df_train, backward_df_testval = add_duplicates_to_training_set(backward_df)
go_df_train, go_df_testval = add_duplicates_to_training_set(go_df)
visual_df_train, visual_df_testval = add_duplicates_to_training_set(visual_df)


# In[18]:


# Stack test_validation dataframe
test_val_array = np.vstack(eval(df).values for df in df_testval_sublist)
test_val_array = test_val_array[:, :-1]


# In[19]:


# Splitting Test and Validation Sets
X_val, X_test, y_val, y_test = train_test_split(test_val_array[:,:-1], test_val_array[:,-1], test_size = 0.67)


# In[20]:


# #Creating test and validation sets
val_set = np.column_stack((X_val,y_val))
test_set = np.column_stack((X_test, y_test))


# In[21]:


# train_set


# # Applying Feature Extraction

# In[22]:


def python_speech_mfcc(raw_train_data, label):
    label_vector = []
    dirname = os.getcwd()
    for index, row in raw_train_data.iterrows():
        (rate,sig) = wavfile.read(dirname + "/" + row['Label'] + "/" + row['FileName'])
        mfcc_feat = mfcc(sig,rate,nfft=1024)
        label_vector.append(mfcc_feat)
        gc.collect
    label_array = np.concatenate(label_vector, axis=0)
    b = [label for i in range(label_array.shape[0])]
    label_array = np.column_stack((label_array, b))
#     print(raw_train_data.head(),label)
#     print(label_array.shape)
    return label_array


# In[23]:


# #Test_Val Set MFCC
# mfcc_df_testval = np.empty((0,14))

# for df, label in zip(df_testval_sublist, labels):
#     mfcc_df_testval = np.append(mfcc_df_testval, python_speech_mfcc(eval(df), label), axis=0)
#     gc.collect


# In[24]:


# #Training Set MFCC
# mfcc_df_train = np.empty((0,14))

# for df, label in zip(df_train_sublist, labels):
#     mfcc_df_train = np.append(mfcc_df_train, python_speech_mfcc(eval(df), label), axis=0)
#     gc.collect


# In[25]:


# save('mfcc_testval.npy', mfcc_df_testval)
# save('mfcc_train.npy', mfcc_df_train)


# In[26]:


# mfcc_df_testval = load('mfcc_testval_clean.npz')
# mfcc_df_train = load('mfcc_train_clean.npz')


# In[29]:


mfcc_df_train = load('mfcc_train.npy')
mfcc_df_train
# X_train = mfcc_df_train['arr_0']
# y_train = mfcc_df_train['arr_1']


# In[84]:





# In[30]:


#Setting X-train and y-train
X_train = mfcc_df_train[:, :-1]
y_train = mfcc_df_train[:, -1]

# Shuffling the training set
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train, random_state=0)


# In[35]:


# #Encoding the labels
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
lenc = LabelEncoder()
y_train = lenc.fit_transform(y_train)
y_train


# In[57]:





# In[9]:


# # One Hot Encoding cause multiclass classification
# from keras.utils import np_utils
# y_train = np_utils.to_categorical(y_train, num_classes=len(labels))


# In[18]:


# savez('mfcc_testval_clean', mfcc_df_testval)
# savez('mfcc_train_clean', X_train, y_train)


# In[38]:


print(type(X_train))
print(type(y_train))


# # Part 1 - GMM-MFCC

# In[8]:


from joblib import parallel_backend, Parallel, delayed
# with parallel_backend('dask.distributed', scheduler_host='scheduler-address:8786'):
    # your now-cluster-ified sklearn code here
    
import os
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6


### Dask-learn pipeline and GridSearchCV drop-in replacements
import dask_searchcv as dcv
import numpy as np
np.__config__.show()
# from sklearn.grid_search import GridSearchCV
# from dklearn.grid_search import GridSearchCV
# # from sklearn.pipeline import Pipeline
# from dklearn.pipeline import Pipeline


# In[41]:


# X_train = data.reshape(-1,1)


# In[56]:


reverse
y_train.shape


# In[41]:

import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA

pca = PCA(0.85, whiten=True)
data = pca.fit_transform(X_train)

n_components = np.arange(750, 2250, 750)
models = [GMM(n, covariance_type='spherical', random_state=0).fit(data)
          for n in n_components]

fig, gmm = plt.subplots(figsize=(10, 6))
bics = [ model.fit(data).bic(data) for model in models ]
# aics = [ model.fit(data).aic(data) for model in models ]
plt.plot(n_components, bics, label='AIC')
print("here")
gmm.legend(loc='best')
gmm.set_xlabel('n_components')
gmm.set_xticks(np.arange(0, 120, 4));


# In[42]:


"""
GMM Bayes
---------
This implements generative classification based on mixtures of gaussians
to model the probability density of each class.
"""

import warnings
import numpy as np

from sklearn.naive_bayes import BaseNB
from sklearn.mixture import GaussianMixture


class GMMBayes(BaseNB):
    """GaussianMixture Bayes Classifier

    This is a generalization to the Naive Bayes classifier: rather than
    modeling the distribution of each class with axis-aligned gaussians,
    GMMBayes models the distribution of each class with mixtures of
    gaussians.  This can lead to better classification in some cases.

    Parameters
    ----------
    n_components : int or list
        number of components to use in the GaussianMixture. If specified as
        a list, it must match the number of class labels. Default is 1.
    **kwargs : dict, optional
        other keywords are passed directly to GaussianMixture
    """

    def __init__(self, n_components=1, **kwargs):
        self.n_components = np.atleast_1d(n_components)
        self.kwargs = kwargs


    def fit(self, X, y):
        X = np.asarray(X).astype("float32")
        y = np.asarray(y).astype("float32")

        n_samples, n_features = X.shape

        if n_samples != y.shape[0]:
            raise ValueError("X and y have incompatible shapes")

        self.classes_ = np.unique(y)
        self.classes_.sort()
        unique_y = self.classes_

        n_classes = unique_y.shape[0]

        if self.n_components.size not in (1, len(unique_y)):
            raise ValueError("n_components must be compatible with "
                             "the number of classes")

        self.gmms_ = [None for i in range(n_classes)]
        self.class_prior_ = np.zeros(n_classes)

        n_comp = np.zeros(len(self.classes_), dtype=int) + self.n_components
        gc.collect()
        for i, y_i in enumerate(unique_y):
            if n_comp[i] > X[y == y_i].shape[0]:
                warnstr = ("Expected n_samples >= n_components but got "
                           "n_samples={0}, n_components={1}, "
                           "n_components set to {0}.")
                warnings.warn(warnstr.format(X[y == y_i].shape[0], n_comp[i]))
                n_comp[i] = y_i
            self.gmms_[i] = GaussianMixture(n_comp[i], **self.kwargs).fit(X[y == y_i])
            self.class_prior_[i] = np.float(np.sum(y == y_i)) / n_samples
            gc.collect()
        return self

    def _joint_log_likelihood(self, X):
        
        X = np.asarray(np.atleast_2d(X), dtype='float32')
        logprobs = np.array([g.score_samples(X) for g in self.gmms_], dtype='float32').T
        return logprobs + np.log(self.class_prior_)


# In[ ]:
data_32_float = np.array(data, dtype='float32')

# Fit the GMMNaive Bayes classifier to all original dimensions
gmm_nb = GMMBayes(750) # 128 components per class
gmm_nb.fit(data_32_float, y_train)
gc.collect()
# now predict
y_pred = gmm_nb.predict(data_32_float)

#get completeness score (equivalent to recall)
completeness_score = recall_score(y_train,y_pred, average='weighted')
#get contamination score (equivalent to 1-precision)
contamination_score = (1-precision_score(y_train,y_pred, average='weighted'))

print('Completeness: %f'%completeness_score)
print('Contamination: %f'%contamination_score)


# In[ ]:
filename = 'gmm_model_10_06_750.sav'
pickle.dump(gmm_nb, open(filename, 'wb'))

gmm_nb = GMMBayes(2000) # 128 components per class
gmm_nb.fit(data_32_float, y_train)
gc.collect()
# now predict
y_pred = gmm_nb.predict(data_32_float)

#get completeness score (equivalent to recall)
completeness_score = recall_score(y_train,y_pred, average='weighted')
#get contamination score (equivalent to 1-precision)
contamination_score = (1-precision_score(y_train,y_pred, average='weighted'))

print('Completeness: %f'%completeness_score)
print('Contamination: %f'%contamination_score)


# In[ ]:
    
filename = 'gmm_model_13_06_2000.sav'
pickle.dump(gmm_nb, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)


# In[33]:


np.column_stack((y_train, y_pred))


# In[268]:


NEED TO TEST HERE


# # Part 2 - CNN

# ## Creating spectrogram files for every training file

# In[238]:


def graph_spectrogram(raw_train_data, label):
    dirname = os.getcwd()
    pathlib.Path(dirname + "/" + label + '/images').mkdir(parents=True, exist_ok=True)
    fig,ax = plt.subplots(1)
    for index, row in raw_train_data.iterrows():
        samplingFrequency, signalData = wavfile.read(dirname + "/" + row['Label'] + "/" + row['FileName'])
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
        ax.axis('off')
        pxx, freqs, bins, im = ax.specgram(x=signalData, Fs=samplingFrequency, noverlap=240, NFFT=512, cmap='viridis')
        ax.axis('off')
        fig.savefig(dirname + "/" + label + '/images/' + row["FileName"] +'.png', dpi=300, frameon='false', transparent=True)
        plt.cla()
    plt.close(fig)
        
for df, label in zip(df_train_sublist, labels):
    graph_spectrogram(eval(df), label)


# In[249]:


#Applying the same to test array
dirname = os.getcwd()
for label in labels:
    pathlib.Path(dirname + "/" + 'test/' + label).mkdir(parents=True, exist_ok=True)
fig,ax = plt.subplots(1)
for row in test_set:
    samplingFrequency, signalData = wavfile.read(dirname + "/" + row[1] + "/" + row[0])
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('off')
    pxx, freqs, bins, im = ax.specgram(x=signalData, Fs=samplingFrequency, noverlap=240, NFFT=512, cmap='viridis')
    ax.axis('off')
    fig.savefig(dirname + "/" + 'test/' + row[1] + '/'+ row[0] +'.png', dpi=300, frameon='false', transparent=True)
    plt.cla()
plt.close(fig)


# In[250]:


#Applying the same to val array
dirname = os.getcwd()
for label in labels:
    pathlib.Path(dirname + "/" + 'val/' + label).mkdir(parents=True, exist_ok=True)
fig,ax = plt.subplots(1)
for row in val_set:
    samplingFrequency, signalData = wavfile.read(dirname + "/" + row[1] + "/" + row[0])
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('off')
    pxx, freqs, bins, im = ax.specgram(x=signalData, Fs=samplingFrequency, noverlap=240, NFFT=512, cmap='viridis')
    ax.axis('off')
    fig.savefig(dirname + "/" + 'val/' + row[1] + '/'+ row[0] +'.png', dpi=300, frameon='false', transparent=True)
    plt.cla()
plt.close(fig)


# #### In the first instance, the data was passed to the CNN by means of the ImageDataGenerator function. In order to make use of better processing power through Google Colab and hence perform more experiments, this was then replaced by the Colab variant explained further on.

# In[39]:


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255, # rescale all pixel values from 0-255, so aftre this step all our pixel values are in range (0,1)
        shear_range=0.2, #to apply some random tranfromations
        zoom_range=0.2,#to apply zoom
        horizontal_flip=True) # image will be flipper horiz
test_datagen = ImageDataGenerator(rescale=1./255)

dirname = os.getcwd()

img_height, img_width = (64,64)

training_set = train_datagen.flow_from_directory(
        dirname,
        target_size=(img_height, img_width),
        batch_size=32,
        color_mode = 'grayscale',
        class_mode='categorical',
        classes = list(labels),
        shuffle = True)

validation_set = test_datagen.flow_from_directory(
        dirname+'/val/',
        target_size=(img_height, img_width),
        batch_size=32,
        color_mode = 'grayscale',
        class_mode='categorical',
        classes = list(labels))

testing_set = test_datagen.flow_from_directory(
        dirname+'/test/',
        target_size=(img_height, img_width),
        batch_size=32,
        color_mode = 'grayscale',
        class_mode='categorical',
        shuffle = True)


# In[43]:


from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from keras import backend as K
K.clear_session()


#Define CNN Model
model = Sequential()
input_shape = (img_height, img_width, 1)
num_classes = 10
batch_size = 32
nb_epochs = 20
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

#Compile Model
model.compile(optimizers.rmsprop(lr=0.0005, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
print(model.summary())

model.fit_generator(
    training_set,
    steps_per_epoch = training_set.samples // batch_size,
    validation_data = validation_set, 
    validation_steps = validation_set.samples // batch_size,
    epochs = nb_epochs)

our_model = model
our_model.save('dirname')


# In[ ]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In order to attempt more experiments using Google Colab, mounting the spectrogram image files and passing them through the imagedatagenerator was proving to be too slow, so we opted to transform dataset into h5 files, reading the images as numpy arrays, and passing these to the notebook on the cloud.

# In[254]:


def normalize_and_write_data_into_h5_file(dest_filepath, filepaths_list, n_px, n_channels):
  
    data_shape = (len(filepaths_list), n_px * n_px * n_channels)
    dataset_name = "input_data"
    
    with h5py.File(dest_filepath, 'a') as f:
        
        f.create_dataset(dataset_name, data_shape, np.float32)
        
        for i in range(len(filepaths_list)):
            #if (i+1) % 512 == 0:
            #    print('{}/{} files converted'.format((i+1), len(filepaths_list)))

            filepath = filepaths_list[i]
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (n_px, n_px), interpolation=cv2.INTER_CUBIC)
            
            #Normalize the image - convert the each pixel value between 0 and 1
            img = img / 255
            #Reshape the image - roll it up into a column vector
            img = img.ravel()
            
            #img[None] makes it a proper array instead of rank 1 array
            f[dataset_name][i, ...] = img[None]


# In[265]:


def write_labels_into_h5_file(dest_filepath, new_labels):
    
    dataset_name = "input_labels"
    
    with h5py.File(dest_filepath, 'a') as f:
        f.create_dataset(dataset_name, (len(new_labels),), np.int8)
        f[dataset_name][...] = new_labels
        
def convert_images_to_data_in_h5_file(src_img_filepath_pattern, dest_h5_file_path, n_px, 
                                      n_channels = 3, batch_size = 1024):
    full_filep = np.empty((0,2))
    for label in labels:
        temp_filepaths = glob.glob(dirname+'/test/'+label+'/*.png') #Use this for test set
#         temp_filepaths = glob.glob(dirname+'/val/'+label+'/*.png') #Use this for validation set
#         temp_filepaths = glob.glob(dirname+ '/' + label+'/images/*.png') #Use this for training set
        label_list= [label] * len(temp_filepaths)
        temp_array = np.column_stack((temp_filepaths, label_list))
        full_filep = np.append(full_filep, temp_array, axis=0)

    np.random.shuffle(full_filep)
#     print(full_filep)
    
    le = preprocessing.LabelEncoder()
    full_filep[:,1] = le.fit_transform(full_filep[:,1])
    full_filepaths = list(full_filep[:,0])
    new_labels = list(full_filep[:,1].astype(int))
    t = list(zip(full_filepaths, new_labels))
    shuffle(t)
    #Get the shuffled filepaths & labels
    src_filepaths, new_labels = zip(*t)

#     print(full_filepaths)
    
    #Number of images
    m = len(src_filepaths)
#     print(m)
    n_complete_batches = math.ceil(m / batch_size)
#     print(n_complete_batches)
    
    for i in range(n_complete_batches):
        print('Creating file', (i+1))
        
        dest_file_path = dest_h5_file_path + str(i + 1) + ".h5"   
        
        start_pos = i * batch_size
        end_pos = min(start_pos + batch_size, m)
        src_filepaths_batch = src_filepaths[start_pos: end_pos]
        labels_batch = new_labels[start_pos: end_pos]
#         print(labels_batch)
        normalize_and_write_data_into_h5_file(dest_file_path, src_filepaths_batch, n_px, n_channels)
        write_labels_into_h5_file(dest_file_path, labels_batch)


# In[266]:


#Specifying the directory name, image size, and no of channels (1 for grayscale)
src_filepath_pattern = dirname
dest_filepath = dirname+'Assignmenttesth5'
n_px = 64
n_channels = 1

convert_images_to_data_in_h5_file(src_filepath_pattern, dest_filepath, n_px, n_channels)


# ## The below section was run in Google Colab.

# In[110]:


import time
import os
import h5py
import matplotlib.pyplot as plt
# os.chdir('/content/drive/My Drive/Speech Assignment')


# In[ ]:


#Loading the h5 datafiles
def load_dataset(prefix, filelimit):
    
    lmd_tic = time.time()
    
    X_full_dataset = []
    Y_full_dataset = []
    filename_prefix = prefix
    
    for i in range(1,filelimit):
        
        filename = filename_prefix + str(i) + ".h5"
        with h5py.File(filename, "r") as f:
    
            X_full_dataset.append(f["input_data"][:])
            Y_full_dataset.append(f["input_labels"][:])

    lmd_toc = time.time()
    print('Time taken to load the data set is', ((lmd_toc-lmd_tic) * 1000), 'ms')
    
    return X_full_dataset, Y_full_dataset


# In[ ]:


#Creating the datasets
X_train_dataset, Y_train_dataset = load_dataset('Assignmenttrainh5', 16)
X_val_dataset, Y_val_dataset = load_dataset('AssignmentAssignmentvalh5', 3)
X_test_dataset, Y_test_dataset = load_dataset('AssignmentAssignmenttesth5', 5)


# In[ ]:


#Formatting the dataset from the H5 File
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()

#Reshaping and encoding the datasets

#Training Set
X_train = np.vstack(X_train_dataset)
Y_train = np.hstack(Y_train_dataset)

Y_train = Y_train.reshape(-1,1)
enc.fit(Y_train)
Y_train = enc.transform(Y_train)
Y_train = Y_train.todense()

#Validation Set
X_val = np.vstack(X_val_dataset)
Y_val = np.vstack(Y_val_dataset)

Y_val = Y_val.reshape(-1,1)
enc.fit(Y_val)
Y_val = enc.transform(Y_val)
Y_val = Y_val.todense()

#Test Set
X_test = np.vstack(X_test_dataset)
Y_test = np.vstack(Y_test_dataset)

Y_test = Y_test.reshape(-1,1)
enc.fit(Y_test)
Y_test = enc.transform(Y_test)
Y_test = Y_test.todense()


# In[ ]:


#Sanity checking types/shapes of vectors we're passing to CNN
print("Training Set")
print(type(X_train))
print(X_train.shape)
print(type(Y_train))
print(Y_train.shape)
print("---------------------------")
print("Validation Set")
print(type(X_val))
print(X_val.shape)
print(type(Y_val))
print(Y_val.shape)
print("---------------------------")
print("Test Set")
print(type(X_test))
print(X_test.shape)
print(type(Y_test))
print(Y_test.shape)
print("---------------------------")


# ### MNIST CNN

# In[ ]:


from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras import regularizers, optimizers, models
from keras import backend as K
K.clear_session()


# In[ ]:


img_height = 64
img_width = 64

#Define CNN Model
input_shape = (img_height, img_width, 1)
num_classes = 10
batch_size = 32
nb_epochs = 20


# In[ ]:


# MNIST MODEL TRAINING SECTION
model = Sequential()
model.add(Flatten(input_shape=input_shape))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


#Compile Model
model.compile(optimizers.rmsprop(lr=0.0005, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
print(model.summary())

#Train and Test The Model
history_mnist = model.fit(
        x = X_train.reshape(X_train.shape[0], img_height, img_width, 1), y = Y_train,
        steps_per_epoch=2000,
        epochs=40,
        validation_data = (X_val.reshape(X_val.shape[0], img_height, img_width, 1), Y_val),
        validation_steps=800)


# In[ ]:


#Save model to h5 file
model.save("mnist_model_ver2.h5")
print("Saved model to disk")


# In[ ]:


#Load model and test
mnist_model = models.load_model('mnist_model_ver2.h5')
mnist_model.summary()

loss,acc = mnist_model.evaluate(X_test.reshape(X_test.shape[0], img_height, img_width, 1),  Y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# In[ ]:


#Plot accuracy vs val_accuracy

plt.plot(history_mnist.history['accuracy'])
plt.plot(history_mnist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# ### CNN Model

# In[ ]:


#Define CNN Model
model2 = Sequential()
input_shape = (img_height, img_width, 1)
num_classes = 10
batch_size = 32
nb_epochs = 20
model2.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))
model2.add(Flatten())
model2.add(Dense(128, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(num_classes, activation='softmax'))

#Compile Model
model2.compile(optimizers.adam(),loss="categorical_crossentropy",metrics=["accuracy"])
print(model2.summary())

#Train and Test The Model
cnn_model_history = model2.fit(
        x = X_train.reshape(X_train.shape[0], img_height, img_width, 1), y = Y_train,
        batch_size = 20,
        epochs=30,
        verbose = 1,
        validation_data = (X_val.reshape(X_val.shape[0], img_height, img_width, 1), Y_val))


# In[ ]:


#Save model to file
model2.save("cnn_model_with_adam_ver2.h5")
print("Saved model to disk")


# In[ ]:


#CNN model load and evaluation
cnn_model = models.load_model('cnn_model_with_adam_ver2.h5')
cnn_model.summary()

loss,acc = cnn_model.evaluate(X_test.reshape(X_test.shape[0], img_height, img_width, 1),  Y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# In[ ]:


#Plot accuracy vs val_accuracy

plt.plot(cnn_model_history.history['accuracy'])
plt.plot(cnn_model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# ### CNN Model #3

# In[ ]:


#Define CNN Model
model3 = Sequential()
input_shape = (img_height, img_width, 1)
num_classes = 10
batch_size = 32
nb_epochs = 20

model3.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model3.add(MaxPooling2D(pool_size=(3, 3)))
model3.add(BatchNormalization())
model3.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model3.add(MaxPooling2D(pool_size=(3, 3)))
model3.add(BatchNormalization())
model3.add(Flatten())
model3.add(Dense(64, activation='relu'))
model3.add(BatchNormalization())
model3.add(Dropout(0.2))
model3.add(Dense(num_classes, activation='softmax'))

#Compile Model
model3.compile(optimizers.adam(),loss="categorical_crossentropy",metrics=["accuracy"])
print(model3.summary())

#Train and Test The Model
history = model3.fit(
        x = X_train.reshape(X_train.shape[0], img_height, img_width, 1), y = Y_train,
        batch_size = 32,
        epochs=100,
        verbose = 1,
        validation_data = (X_val.reshape(X_val.shape[0], img_height, img_width, 1), Y_val))


# In[ ]:


#Saving model to disk
model3.save("cnn_model_batchnorm_ver2.h5")
print("Saved model to disk")


# In[ ]:


#Loading and evaluation
cnn_model = models.load_model("cnn_model_batchnorm_ver2.h5")
cnn_model.summary()

loss,acc = cnn_model.evaluate(X_test.reshape(X_test.shape[0], img_height, img_width, 1),  Y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# In[ ]:


#Plotting Accuracy versus Validation Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

