#Impliment AI powered chatbot application using NLTK, Keras and Tkinter.
#Implimenting necessary libraries

import nltk    #Natural language toolkit for text processing
from keras.src.saving import load_model
from nltk.stem import WordNetLemmatizer #WordNetLemmatizer: Used to reduce words to their base form (e.g., “running” → “run”)
lemmatizer = WordNetLemmatizer()
import pickle   #pickle: To load saved Python objects (words.pkl, classes.pkl)
import numpy as np #Used for array

import tensorflow.keras.models
model = load_model('chatbot.h5')  #Loads a pre-trained Keras model, chatbot model.
import json
import random
intents = json.loads(open('intents.json').read())   #intents json with tags, patterns and responses.
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def clean_up_sentence(sentence):    #tokenize the pattern, split words into an array
    sentence_words = nltk.word_tokenize(sentence)       #stem each word - create short form for word.
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)  #Tokenize the pattern
    #bag of words- matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return (np.array(bag))

def predict_class(sentence, model):
    #filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p])) [0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    #sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list

def getresponse(ints, intents_json):
    tag = ints[0][0]
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag) :
            result = random.choice(i['responses'])
            break
        else:
            result = "Sorry, I dont understand that."
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    if not ints:
        return "Sorry, I don't understand that."
    res = getresponse(ints, intents)
    return res

 # creating GUI with Tkinter
import tkinter
from tkinter import *

def send(): # this function is used in a Tkinter based chatbot GUI, and triggered when user clicks send button.
    msg = EntryBox.get("1.0", 'end-1c').strip() 
    EntryBox.delete("0.0", END)
    
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: "+ msg + '\n\n')
        ChatLog.config(foreground="#442265", font =("verdana"))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

base = Tk()
base.title("Chatbot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

#Create chat window
ChatLog = Text(base, bg="white", height ="8", width = "50", font= "Ariel",)
ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command = ChatLog.yview, cursor = "heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create button to send message
SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12" , height=5, bd=0, bg="#25cdf7",
activebackground="#3c9d99", fg="#3c9d99", command = send )

#Create the box to enter message
EntryBox = Text(base, bd=0,bg ="white", width ="29" , height ="5", font= "Ariel",)

#Place all components on the screen
scrollbar.place(x=376, y=6, height =386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()







