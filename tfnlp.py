import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences

sentences = ["I love my dog","I love my cat","You love my dog!","Do you think my dog is amazing?"]

tokenizer = Tokenizer(num_words=100,oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index #Gives index for each word
print(word_index)

for i in word_index:
    print(i)
#sequences = tokenizer.texts_to_sequences(sentences) #prints each sentence is word index squences
#print(sequences)
#padded = pad_sequences(sequences)
#print(padded)


