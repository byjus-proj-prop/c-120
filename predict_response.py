import nltk
import json
import pickle
import numpy as np
import random
import tensorflow
import data_preprocessing

ignore_words = ["?", "!", ","];
model = tensorflow.keras.models.load_model("chatbot_model.h5");
intents =  json.loads(open("intents.json").read());
words = pickle.load(open("words.pkl", "rb"));
classes = pickle.load(open("classes.pkl", "rb"));

def input_proccess(text):
    input_word1 = nltk.word_tokenize(text);
    input_word2 = data_preprocessing.get_stem_words(input_word1, ignore_words);
    input_word2 = sorted(list(set(input_word2)));
    bag = [];
    bag_of_words = [];
    for word in words:
        if word in input_word2: 
            bag_of_words.append(1);
        else: bag_of_words.append(0);
    bag.append(bag_of_words)
    return np.array(bag);
def bot_class_prediction(text):
    inp = input_proccess(text);
    prediction = model.predict(inp);
    predicted_class_label = np.argmax(prediction[0]);
    return predicted_class_label;
def bot_response(text):
    predicted_class_label = bot_class_prediction(text);
    predicted_class = classes[predicted_class_label];
    for intent in intents["intents"]:
        if intent["tag"] == predicted_class:
            bot_response = random.choice(intent["responses"]);
            return bot_response
while True:
    user_input = input("Digite aqui sua mensagem: ");
    print("Entrada do usuario: ", user_input);
    response = bot_response(user_input);
    print("Resposta: ", response);