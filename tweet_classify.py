# Import the libraries

import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np



# Load the dataframe

print("Loading the dataset")



df = pd.read_csv('train.csv')
df.head()




# Shape of the original data

print("Shape of original dataset",df.shape)



# ### Features use
# 
# 
# * Mail address
# * date of birth / age
# * Bank details
# * Phone number
# * Address

# ## Preprocessing the data
 
# ## Process on all tweet


print("Preprocessing all the tweet")
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup

# start the wordtokenizer instance
tok = WordPunctTokenizer()


# regex for extracting @tag
at_rate = r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)'

# regex for extract url link
url_link = r'https?://[A-Za-z0-9./]+'

# regex for hashtag
hashtag = r'(\s)#\w+'

# Combine both regex
combine = r'|'.join((at_rate, url_link))

def preprosessing(text):
    
    # parse text in lxml format
    soup = BeautifulSoup(text, 'lxml')
    
     # extract the text
    new_text = soup.get_text()
    
    # subtract the @tag and url link
    strip_text = re.sub(combine, '', new_text)
    
    
    # remove hashtag
    strip_text = re.sub(hashtag, r'', strip_text)
    
    
    try:
        # remove all non-ascii character 
        clean = ''.join([i if ord(i) < 128 else ' ' for i in strip_text])
        
    except:
        clean = strip_text
        
    # convert all text to lower case
    lower_case = clean.lower()
    
    # remove all special characters
    lower_case = re.sub(r'[()\"#/;:<>{}`+=~|!?,]', '', lower_case)
    
    
    # tokenize and join together to remove unneccessary white spaces
    words = lower_case.split()
    
    # remove unnecessay white spaces,
    return (" ".join(words)).strip()


# ## Process all tweets

# In[14]:


# Make another column for processed tweet

df['processed_tweet'] = df['tweet'].apply(preprosessing)

#'''


#   df = pd.read_csv('train_processed.csv')



# Make 2 new columns

df['is_Threat_user'] = 0
df['subject'] = None


# #### 1.Email

# In[16]:

print("1. extracting email features")

count = 0
for row_no,tweet in enumerate(df['processed_tweet']):
    
    # regex for extracting email feature
    if re.findall("[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",str(tweet)):
        
        #print(row_no,tweet)
        
        # change is_Threat_user column value to 1
        df.loc[row_no, 'is_Threat_user'] = 1
        
        # change subject to email so that we can identified the feature for "is_Threat_user" column 
        df.loc[row_no, 'subject'] = "Email"
        
        # count the total email features
        count += 1
print("Total Email features extracted : ", count)  


# #### 2.Age / Birth date

# In[17]:

print("2. extracting age features")

count = 0
for row_no,tweet in enumerate(df['processed_tweet']):
    
    # regex for extracting age or birthdate feature
    if (re.findall("(?:date of birth|my dob |b - day|my age |years old)",str(tweet))):
        
        #print(row_no,tweet)
        
        # change is_Threat_user column value to 1
        df.loc[row_no, 'is_Threat_user'] = 1
        
         # change subject to age so that we can identified the feature for "is_Threat_user" column 
        df.loc[row_no, 'subject'] = "Age"
        
        # count the total age features
        count += 1
        
print("Total age features extracted : ", count) 


# #### 3. Bank Account no 

# In[18]:

print("3. extracting Bank details features")
count = 0
for row_no,tweet in enumerate(df['processed_tweet']):
    
    # regex for extracting bank details feature
    if (re.findall("(?:my debit card|my credit card|my account no|my account number|my bank)",str(tweet))):
        
        # print(row_no,tweet)
        
        # change is_Threat_user column value to 1
        df.loc[row_no, 'is_Threat_user'] = 1
        
        # change subject to Bank details so that we can identified the feature for "is_Threat_user" column 
        df.loc[row_no, 'subject'] = "Bank"
        
        # count the total bank details features
        count += 1
        
print("Total bank details features extracted : ", count)  


# #### 4.phone number

# In[19]:

print("4. extracting phone number features")
count = 0
for row_no,tweet in enumerate(df['processed_tweet']):
    
    # regex for extracting phone number features
    if (re.findall(r"[a-z. ]+ \+?1?\s*\(?-*\.*(\d{3})\)?\.*-*\s*(\d{3})\.*-*\s*(\d{4})$",str(tweet))):
        
        #print(row_no,tweet)
        
        # change is_Threat_user column value to 1
        df.loc[row_no, 'is_Threat_user'] = 1
        
        # change subject to Phone so that we can identified the feature for "is_Threat_user" column 
        df.loc[row_no, 'subject'] = "Phone"
        
        # count the total phone details features
        count += 1
        
print("Total Phone number features extracted : ", count)  


# #### 5.Address

# In[20]:

print("5. extracting address features")
count = 0
for row_no,tweet in enumerate(df['processed_tweet']):
    
    # regex for extracting address details feature
    if (re.findall("(?:my location|my current location|my home address|my house address|i live in)",str(tweet))):
        
        #print(row_no,tweet)
        
        # change is_Threat_user column value to 1
        df.loc[row_no, 'is_Threat_user'] = 1
        
        # change subject to Adress details so that we can identified the feature for "is_Threat_user" column 
        df.loc[row_no, 'subject'] = "Address"
        
         # count the total Address details features
        count += 1
        
print("Total Address details features extracted : ", count) 


# In[21]:


# checking shape of all vulnerable user's tweet 

print("Shape of total vulnerable people's tweet :",df[df["is_Threat_user"]==1].shape)


# In[22]:


# see the dataframe
df.head()


# ### Taking Samples

# In[23]:

print("Taking samples from the dataset")
# Taking 2500 samples from normal people tweet
df_0 = df[df['is_Threat_user']==0].sample(n=2500)

# Taking all samples from vulnerable people tweet
df_1 = df[df['is_Threat_user']==1].sample(n=2426)


# In[24]:


# merge the two dataframe
frames = [df_0, df_1]

# concat the dataframe to result 
result = pd.concat(frames)


# In[25]:


result.head()


# In[26]:


# randomly suffle all data by taking all sample at random
result = result.sample(frac=1)


# In[27]:

print("split the datset into training and testing set")

from sklearn.model_selection import train_test_split

# split the datset into two set, training set and 8 percentage testing set
train_df, test_df = train_test_split(result, test_size=0.08)


# In[28]:


print("Shape of training phase data : ",train_df.shape)
print("Shape of testing phase data : ",test_df.shape)


# In[29]:


train_df.head()


# In[30]:


# Changing dataframe index to user name
test_df.index = test_df.name


# In[31]:


test_df.head()


# # Modeling

# In[32]:

print("split the dataset into traing set and validation set")
from sklearn.model_selection import train_test_split

# Again split the training set to training set (80%) and validation set (20%)

# X_train ->  features of training set
# y_train ->  label of training set

# X_test ->  features of validation set
# y_test ->  label of validation set

X_train, X_test, y_train, y_test = train_test_split(train_df['processed_tweet'], train_df['is_Threat_user'], test_size=0.20, random_state=42)


# In[33]:


print("Training set shape : ",X_train.shape)
print("Validation set shape : ",X_test.shape)


# In[34]:

print("Transform the features for Modeling")
from sklearn.feature_extraction.text import CountVectorizer

# start the instance of CountVectrizer which convert word to vector
count_vect = CountVectorizer()


# In[35]:


# converting all words in training dataset to vector form

X_train_counts = count_vect.fit_transform(X_train)

print("Shape of input features in training data : ",X_train_counts.shape)


# In[36]:


from sklearn.feature_extraction.text import TfidfTransformer

# start the instance of TfidfTransformer which get word occurance frequency in each documents
tfidf_transformer = TfidfTransformer()

# Transform all the documents to Inverse Document Frequency matrix representation  
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

print("Shape of input feature matrix in training data : ",X_train_tfidf.shape)


# ### Building the SVM model

# ### Apply the model

# In[38]:

print("Applying the model")
# Building the pipeline
from sklearn.pipeline import Pipeline

# import the classifier
from sklearn.linear_model import SGDClassifier

# make the svm classifier with pipeline
# make (1,2) n-gram word vectorizer
# make tfidf transformer to false
# alpha = learning rate


svm_clf = Pipeline([('vect', CountVectorizer( ngram_range=(1, 2))),
                      ('tfidf', TfidfTransformer( use_idf=False)),
                      ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-4, max_iter=10, random_state=42))])


# In[39]:


# Fit the data in classifier
_ = svm_clf.fit(X_train, y_train)

# predict the validation set
predicted_svm = svm_clf.predict(X_test)

# print accuracy
print("Accuracy of svm classifier on validation set : ",np.mean(predicted_svm == y_test))


# #### Finding optimum Parameters using gridsearch method

# In[40]:

print("Extracting the optimum parameters")

from sklearn.model_selection import GridSearchCV

# Range of parameters to given fo the classifier
parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)],'tfidf__use_idf': (True, False),'clf-svm__alpha': (1e-2, 1e-3)}

# define the model
grid_svm = GridSearchCV(svm_clf, parameters_svm, n_jobs=-1)


# In[41]:


# fit the model
_ = grid_svm.fit(X_train, y_train)

print("Accuracy : ")
print(grid_svm.best_score_)

print("Optimum Parameters :")
print(grid_svm.best_params_)


# ## Confusion Matrix

# In[42]:

print("Confusion matrix is : ")
from sklearn.metrics import confusion_matrix

import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, predicted_svm)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
class_names = ["Not in Risk","In Risk"]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show(block=False)
#plt.pause(0.001)
#input("Press [enter] to continue.")




# # Evaluation on test data

# In[43]:
print("Predictionon on test datset")


test_tweet = test_df.processed_tweet.tolist()
print("Total test data :",len(test_tweet))


# In[44]:


test_df.to_csv("prediction.csv",index=False)
print("Predict tweet of one of these users : ")
print(test_df['name'].tolist()[:100])


# ### Predict on new data

# In[49]:


# convert word to vector
X_new_counts = count_vect.transform(test_tweet)

# convert word count frequncy matrix
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

# predict for the test data
predicted =svm_clf.predict(test_tweet)

# make a column for predicted value
test_df['predicted'] = predicted

# list of original label 
test_y = test_df['is_Threat_user'].tolist()


# In[50]:


# Accuracy on test set

print("Accuracy on test set : ",np.mean(predicted == test_y))


# ## Misclassified tweet

# In[51]:


print("Total misclassified tweet :",test_df[test_df['is_Threat_user'] != test_df['predicted']].shape)


# In[52]:


mis_df = test_df[test_df['is_Threat_user'] != test_df['predicted']]


# #### misclassified invulnerable person's tweet

# In[53]:


print("misclassified invulnerable person's tweet")


# In[54]:


for i in mis_df[mis_df['predicted']==1]['tweet'].tolist():
    print(i)


# #### misclassified vulnerable person's tweet

# In[55]:


for i in mis_df[mis_df['predicted']==0]['tweet'].tolist():
    print(i)


# ## Prediction on new dataset

# In[56]:


print(test_df.head(20))


# In[60]:
print("**************************************")
print("\\n\nCheck if user at risk or not\n\n")
def result(person):
    
    tweet = [test_df.loc[person]['tweet']]
    
    X_new_counts = count_vect.transform(tweet)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predicted =svm_clf.predict(tweet)
    
    print(" <{}> user tweet this ---> \n\n{}".format(person,test_df.loc[person]['tweet']))
    
    print("\n*****************************")
    
    if predicted==1:
        print("<{}> user is in risk".format(person))
    else:
        print("<{}> user is not in risk".format(person))
    return None


user = input("Enter the username : ")

result(user)

