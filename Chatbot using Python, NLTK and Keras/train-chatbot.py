#Train testing data

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json # to read the intents chatbot file
import pickle # to save and load python objects like word and classes

import numpy as np #for Numerical Array
from tensorflow.keras.models import Sequential # Used to build a linear stack of layers for the neural network
from tensorflow.keras.layers import Dense, Activation, Dropout   #Dense and dropout to define layers of neural network
from tensorflow.keras.optimizers import SGD # Stochastic Gradient Descent optimizer
import random #Random is used to shuffle training data

#words: holds all tokenized words. #classes: holds unique intent tags.
#documents: list of pairs – each pattern and its associated tag. ignore_words: punctuation to ignore.
words =[]
classes = []
documents = []
ignore_words = ['?','!']
data_file = open('intents.json').read()
intents =json.loads(data_file)
# Open and reads Json file and converts Json string into a python dictionary.

for intent in intents['intents']:
    for pattern in intent["patterns"]:
        w = nltk.word_tokenize(pattern)   #tokanize the sentence
        words.extend(w)                   #add words to vocabulary
        documents.append((w, intent['tag']))  #store(words,tag)
        if intent['tag'] not in classes:
            classes.append(intent['tag'])    #store unique tags


# clean and finalize word list
#lemmatize and lowercase all words, removing ignored characters. remove duplicates using set() and sort alphabetically.
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

#Print Info
print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

#Save words and classes
#Save words and classes to .pkl files so you can use them later during inference
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#Build training Data Training - list of input/output pairs, output_empty [0,0,1,0] if the third class is correct one
training = []
output_empty = [0] * len(classes)

# Create Bag of words and output vector
# Lemmatize it. create a bag-of-words vector with 1a where known words appear.
#create output vector with 1 at the correct intent index. add bag,output to training data.
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
#    print(len(bag), len(output_row))
    training.append([bag,output_row])

#shuffle and prepare Data
#shuffle the training data for randomness. convert into Numpy arrays.
# train_x: input features(bag of words), train_y : output labels(intent vectors)
random.shuffle(training)
#training = np.array(training)
train_x = []
train_y = []

for t in training:
    train_x.append(t[0])
    train_y.append(t[1])
#train_x = list(training[:,0])
#train_y = list(training[:,1])


train_x = np.array(train_x)
train_y = np.array(train_y)
print("Training Data Created")

#Build the Neural Network
# A 3 layer neural network. Input 128 neurons, ReLu activation. Dropout - to prevent overfitting
#Hidden layer : 64 neurons
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(len(train_x[0]),)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#Compile the model
#used SGD with Nesterov momentum. Accuracy will be tracked.
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#Train and Save the model
#Train the model for 200 epochs with batch size5. saves the trained model to a file chatbot_model.h5
hist =model.fit(np.array(train_x), np.array(train_y), batch_size=5, epochs=200, verbose=1)
model.save('chatbot.h5',hist)

print("Model created and saved")