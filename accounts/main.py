import numpy as np
import pandas as pd
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, SpatialDropout1D
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import nltk
import keras
import pickle

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    load_tokenizer = pickle.load(handle)

# Load the model
load_model = keras.models.load_model('Hate & Offense_model.h5')
# Specify the path to the NLTK data directory
nltk.data.path.append("D:\Internship Luminar\Offensive\Jeslin rani NLP")

# Download the 'stopwords' corpus
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

df_twitter = pd.read_csv("D:/Internship Luminar/Offensive/Jeslin rani NLP/train.csv")
df_offensive = pd.read_csv("D:/Internship Luminar/Offensive/Jeslin rani NLP/labeled_data.csv")
df_offensive.drop(['Unnamed: 0', 'count', 'hate_speech', 'offensive_language', 'neither'], axis=1, inplace=True)
df_offensive['class'].replace({0: 1}, inplace=True)
df_offensive['class'].replace({2: 0}, inplace=True)
df_offensive.rename(columns={'class': 'label'}, inplace=True)
frame = [df_twitter, df_offensive]
df = pd.concat(frame)

test_samples = [
    "This is a positive sentence.",
    "I love this product!",
    "I hate spam emails.",
    "The weather is nice today.",
]
ytest = [0, 1, 1, 0]

def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word.lower() not in stop_words]
    text = " ".join(text)
    return text

df['tweet'] = df['tweet'].apply(clean_text)
df.to_csv('test.csv', index=False)

stop_words = set(stopwords.words('english'))

def make_wordcloud(df):
    comment_words = ""
    for val in df.tweet:
        val = str(val).lower()
        comment_words += " ".join(val) + " "
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stopwords, min_font_size=10).generate(comment_words)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

df.to_csv('testing.csv', index=False)

x = df["tweet"]
y = df["label"]

max_words = 50000
max_len = 500
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(x)
sequences = tokenizer.texts_to_sequences(x)
sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)

model = Sequential()
model.add(Embedding(max_words, 100, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

stop = EarlyStopping(monitor='val_accuracy',
                     mode='max',
                     patience=5)

checkpoint = ModelCheckpoint(filepath='./',
                             save_weights_only=True,
                             monitor='val_accuracy',
                             mode='max',
                             save_best_only=True)

# Process test samples
xtest = [clean_text(sample) for sample in test_samples]
test_sequences = tokenizer.texts_to_sequences(xtest)

# Pad the test sequences to match the expected input shape
max_len = 500
padded_test_sequences = sequence.pad_sequences(test_sequences, maxlen=max_len)

# Now you can use padded_test_sequences for prediction
pred = model.predict(padded_test_sequences)

# Rest of your code for evaluation and predictions

res = []
for prediction in pred:
    var=prediction[0]*0.01
    if var < 0.05:
        res.append(0)
    else:
        res.append(1)


with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

model.save('Hate & Offense_model.h5')

load_model = keras.models.load_model('./Hate & Offense_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    load_tokenizer = pickle.load(handle)

test = 'you look ReALLy ShittY ToDaY'
test = [clean_text(test)]
print(test)
seq = load_tokenizer.texts_to_sequences(test)
padded = sequence.pad_sequences(seq, maxlen=300)
print(seq)
pred = load_model.predict(padded)
print("pred", pred)
if pred < 0.5:
    print("no hate")
else:
    print("hate and abusive")

