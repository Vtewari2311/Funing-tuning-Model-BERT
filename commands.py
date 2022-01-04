import A3
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from transformers import TFAutoModel
model = TFAutoModel. from_pretrained("distilbert-base-uncased")
# PS C:\Users\guest1> cd '.\AppData\Local\Programs\Python\Python38\Scripts\'
# PS C:\Users\guest1\AppData\Local\Programs\Python\Python38\Scripts> .\pip.exe install tensorflow
# PS C:\Users\guest1\AppData\Local\Programs\Python\Python38\Scripts> .\pip.exe install transformers
from transformers import AutoTokenizer, TFAutoModel
model = TFAutoModel. from_pretrained("distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokens = tokenizer.tokenize("The dog is playing.")
print(tokens)
#token_ids = tokenizer.convert_tokens_to_ids(tokens)
#print(token_ids)
tokens = tokenizer.tokenize("The dog is playing doggygame.")
print(tokens)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)
tokens = ["[CLS]"] + tokens + ["[SEP]"]
print(tokens)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)
# Easier method
# t is a dictionary
t = tokenizer("The dog is playing doggygame.")
print(t)
print(t["input_ids"])
print(t["attention_mask"])
t = tokenizer(["The dog is playing doggygame.", "The cat is sleeping."])
print(t["input_ids"])
t = tokenizer(["The dog is playing doggygame.", "The cat is sleeping."],max_length=9,truncation=True)
print(t)
t = tokenizer(["The dog is playing doggygame.", "The cat is sleeping."],max_length=9,truncation=True,padding=True)
print(t)
print(tokenizer.decode(t["input_ids"][1]))
t = tokenizer(["The dog is playing doggygame.", "The cat is sleeping."],max_length=9,truncation=True,padding=True,return_tensors="tf")
print(t["input_ids"])
print(type(t["input_ids"]))
print(t["attention_mask"])
# output
output = model(t["input_ids"],attention_mask=t["attention_mask"])
print(output[0].shape)
print(output[0][0])
print(output[0][0][0])
print(output[0][0][0].shape)
print(output[0][0][1])
print(output[0][1][0].shape)
print(output[0][0][2])
print(output[0][:,0].shape)

maxTrEg=1000 # maximum number of pos & neg training examples
maxTeEg=1000# maximum number of pos & neg test examples
train_x, train_y = A3.readPosNe("aclImdb/train/pos""aclImdb/train/neg",maxTrEg)
test_x, test_y = A3.readPosNeg("aclImdb/test/pos","aclImdb/test/neg",maxTeEg)
print(train_x[0])
print(train_y[0])
print(test_x[0])
print(test_y[0])
# tokenize train and test set
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenized_train = tokenizer(train_x, max_length=maxlen, 
truncation=True, padding=True, return_tensors="tf")
tokenized_test = tokenizer(test_x, max_length=maxlen, 
truncation=True, padding=True, return_tensors="tf")

# Keras model
bert_model = TFAutoModel.from_pretrained("distilbert-base-uncased")bert_model.trainable = False
    
token_ids = Input(shape=(maxlen,), dtype=tf.int32,
                                      name="token_ids")
attention_masks = Input(shape=(maxlen,), dtype=tf.int32,
                                            name="attention_masks")
bert_output = bert_model(token_ids,attention_mask=attention_masks)output = Dense(2,activation="softmax")(bert_output[0][:,0])
model = Model(inputs=[token_ids,attention_masks],outputs=output)

# Alternate model
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
bert_model = TFAutoModel.from_pretrained("distilbert-base-uncased")
bert_model.trainable = False
    
token_ids = Input(shape=(maxlen,), dtype=tf.int32,
                                      name="token_ids")
attention_masks = Input(shape=(maxlen,), dtype=tf.int32,
                                            name="attention_masks")
bert_output = bert_model(token_ids,attention_mask=attention_masks)
a = Dense(64,activation="relu")(bert_output[0][:,0])
output = Dense(2,activation="softmax")(a)
model = Model(inputs=[token_ids,attention_masks],outputs=output)

print(model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]))
print(model.fit([tokenized_train["input_ids"],tokenized_train["attention_mask"]], train_y, batch_size=25, epochs=3))

# To evaluate the model
score = model.evaluate([tokenized_test["input_ids"],tokeni
zed_test["attention_mask"]], test_y, verbose=0)
print("Accuracy on test data:", score[1])

# Predictions for the model
a = model.predict([tokenized_test["input_ids"], 
tokenized_test["attention_mask"]])
print(a.shape)
print(a[0])
print(a[1])
