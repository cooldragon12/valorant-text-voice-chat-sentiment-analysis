
# <h1 style="text-align:center;">Sentiment Analysis: Valorant Chats</h1>
# This sentiment analysis goals
# ### Import data
import pandas as pd
import scipy 
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import WordNetLemmatizer
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import advertools as adv
# labeled_dataset
import seaborn as sns

# %% [markdown]
# ## Dataset

# %%
uncleaned_dataset = pd.read_csv("./CSIS3112- Natural Language Processing Survey.csv")
uncleaned_dataset.info()

# %%
labeled_dataset = pd.read_csv("./labeled_dataset.csv")
labeled_dataset.info()

# %%
player_experience_dataset = pd.read_csv("./player_experience.csv")
player_experience_dataset.info()

# %% [markdown]
# ### Preprocessing

# %% [markdown]
# 

# For this preprocessing we use nltk and advertools packages to provide stop words for tagalog and english
labeled_dataset['message']=labeled_dataset['message'].apply(lambda x: str(x))

# %%
corpus = []
for i in range(0,len(labeled_dataset)):
    # Remove special characters
    tweet = re.sub('[^a-zA-Z#]', ' ', labeled_dataset['message'][i])
    chat = labeled_dataset['message'][i]
    chat = chat.lower()
    chat = chat.split()
    # Remove stop words
    all_stopwords_eng = stopwords.words('english')
    all_stopwords_eng.remove('not')
    all_stopwords_tag = adv.stopwords['tagalog']
    chat = [word for word in chat if not word in set(all_stopwords_eng) and word not in set(all_stopwords_tag)]
    chat = ' '.join(chat)
    corpus.append(chat)

# %%
labeled_dataset["cl_message"] = corpus
labeled_dataset.tail(10)

# %%
labeled_dataset=labeled_dataset.dropna()

# %%
labeled_dataset.head()

# %%
tokenized_chat= labeled_dataset['cl_message'].apply(lambda x: x.split()) # Rows as x
tokenized_chat.head()

# %%
lem = nltk.WordNetLemmatizer()
tokenized_chat= tokenized_chat.apply(lambda x: [lem.lemmatize(word) for word in x])

# %%
labeled_dataset["cl_message"] = tokenized_chat
labeled_dataset

# %% [markdown]
# ----------------------------------------------

# %% [markdown]
# # Visualizations

# %% [markdown]
# In this section, the visualization ormutaion of the current data will discuss and show.

# %%
def normalization(df, column):
    # Using Z-Score for normalizing the column 
    normal = (df[column] - df[column].mean()) / df[column].std()
    return normal

# %%
servers = player_experience_dataset["most_played_server"].dropna().unique() # Servers in SEA

# %%
most_played_server=player_experience_dataset["most_played_server"].value_counts()


# %%
count_players_agree= player_experience_dataset["is_chat_toxic_than_before"].value_counts()
labels = 'Yes','No'
explode = (0, 0.1)



# %% [markdown]
# ### <b>Regular players vs. Non-Regular</b>

# %%
# regular vs. non_regular
regular_players = player_experience_dataset[player_experience_dataset["is_regular_player"] != "No"]
non_regular_players = player_experience_dataset[player_experience_dataset["is_regular_player"] != "Yes"]


def df_players(r_data, nr_data, index):
    df = pd.DataFrame({"regular":r_data, "non-regular":nr_data}, index=index)
    df = df.fillna(0)
    # df["regular"]=((df["regular"] - df["regular"].mean()) / df["regular"].std())
    # df["non-regular"]=((df["non-regular"] - df["non-regular"].mean()) / df["non-regular"].std()) 

    return df



# %% [markdown]
# ### Often Servers Players Played At

# %%
server_regular = regular_players["most_played_server"].value_counts()
server_non_regular = non_regular_players["most_played_server"].value_counts()

most_played_by_players = df_players(server_regular, server_non_regular, servers)



# %%
toxic_regular = regular_players["toxic_part_of_game"].value_counts().sort_index()
toxic_non_regular = non_regular_players["toxic_part_of_game"].value_counts().sort_index()

toxic_regular = toxic_regular.reindex(["Agent Select", "Early-game", "Mid-game","Late-game"])
toxic_non_regular = toxic_non_regular.reindex(["Agent Select", "Early-game", "Mid-game","Late-game"])
labels_part_game = toxic_regular.index

part_game_toxic_occurs = df_players(toxic_regular, toxic_non_regular, labels_part_game)

# %%
part_game_toxic_occurs["regular"] = part_game_toxic_occurs["regular"]/part_game_toxic_occurs["regular"].sum() 
part_game_toxic_occurs["non-regular"] = part_game_toxic_occurs["non-regular"]/part_game_toxic_occurs["non-regular"].sum()





often_regular = regular_players["often_a_week"].value_counts().sort_index()
often_non_regular = non_regular_players["often_a_week"].value_counts().sort_index()

df_often = df_players(often_regular, often_non_regular, often_regular.index)
df_often


# %% [markdown]
# ### Most toxic server for players

# %%
most_toxic_regular = regular_players["most_toxic_server"].value_counts()
most_toxic_non_regular = non_regular_players["most_toxic_server"].value_counts()

server_toxic = df_players(most_toxic_regular, most_toxic_non_regular, servers)


# %%
text_c = labeled_dataset[labeled_dataset["type"] == "text"]
voice_c = labeled_dataset[labeled_dataset["type"] == "voice"]
text_c_val = text_c["label"].value_counts()
voice_c_val = voice_c["label"].value_counts()
text_c

# %%
def to_string(x):
    if len(x) == 1:
        return x[0]
    elif len(x) == 0:
        return ""
    else:
        return " ".join(x)
labeled_dataset["cl_message_un"] = labeled_dataset["cl_message"].apply(lambda x: to_string(x))




# %%
vouny= labeled_dataset["cl_message_un"].value_counts()
vouny = round((vouny/len(vouny))*100, 1)


labeled_dataset["label"] = labeled_dataset["label"].apply(lambda x: 1 if x == "positive" else -1 if x == "negative" else 0)

labeled_dataset = labeled_dataset.sample(frac = 1)


# %%
tknz = Tokenizer()
tknz.fit_on_texts(labeled_dataset["cl_message_un"])
vectored_count = tknz.texts_to_matrix(labeled_dataset["cl_message_un"], mode='count')
vectored_count

# %% [markdown]
# Using TF-IDF Vectorizer

# %%
tknz = Tokenizer()
tknz.fit_on_texts(labeled_dataset["cl_message_un"])
vectored_tf = tknz.texts_to_matrix(labeled_dataset["cl_message_un"], mode="tfidf")
vectored_tf

# %% [markdown]
# <b>Splitting the dataset</b>
# We splitted the data set 70% for both testing and training set, but the source would be reciprocal from each other.

# %%
y=labeled_dataset["label"].values
x_train, x_test, y_train,y_test = train_test_split(vectored_count,y,test_size=0.3)
print(len(x_train), 'train sequences')
print(len(x_test), 'train sequences')

# %%
print('x_train shape:', x_train.shape) 
print('x_test shape:', x_test.shape)

# %% [markdown]
# ## Using Multinomial Linear Regression

# %%
from sklearn.linear_model import LogisticRegression

# %% [markdown]
# ### Training of Dataset

# %%
lregressor = LogisticRegression(random_state=20, multi_class='multinomial',solver="newton-cg",n_jobs=10, fit_intercept=True, max_iter=1000)
lregressor.fit(x_train, y_train)

# %%
y_lpred = lregressor.predict(x_test)
df = pd.DataFrame({"Predicted": y_lpred,"Actual":y_test})
df

# %%
params = lregressor.get_params()
print(params)

# %%
coef = lregressor.coef_
intercept = lregressor.intercept_
print('Intercept: \n', intercept)
print('Coefficients: \n', coef)

# %%
plt.figure(figsize=(8,8))
plt.plot(intercept)
# plt.move_legend()
# plt.ylabel('Actual label');
# plt.xlabel('Predicted label');
# all_sample_title = 'Accur
# acy Score: {0}'.format(ascore)
plt.title("Intercepts", size = 15);
plt.show()

# %% [markdown]
# Using Metric to Measure Accuracy

# %%
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report,precision_recall_fscore_support

# %%
cm = confusion_matrix(y_test, y_lpred)
print(cm)
ascore = accuracy_score(y_test, y_lpred)
print("\nAccuracy Score of Model Using Multinomial")
print("The Accuracy Score is "+str(ascore))

# %%
class_report_ml=classification_report(y_test, y_lpred,target_names=["negative (-1)","neutral (0)", "positive (1)"], output_dict=True)
class_rep_df_ml = pd.DataFrame(class_report_ml)
class_rep_df_ml

# %%
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r',
xticklabels=["negative","neutral","positive"],yticklabels=["negative","neutral","positive"])
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(ascore)
plt.title(all_sample_title, size = 15)
