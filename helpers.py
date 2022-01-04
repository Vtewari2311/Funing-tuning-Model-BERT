""" Text classification on IMDB dataset by fine-tuning BERT. """
import tensorflow as tf
from A3 import readPosNeg
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from transformers import AutoTokenizer, TFAutoModel

# Class created which combines functions in A3.py with additional functions used to generate models for 
# training and testing purposes of text classification by fine-tuning BERT Models

# Also calculating loss and accuracy + predicitions for models running on smaller dataset

# Only used as a basis of reference
# Model 0:  def train_test_intial_model()
#           def train_test_small_dataset_intial_model()

#
# Model 1:  def train_test()
#           def train_test_small_dataset()

#
# Model 2:  def train_test_alternate_model()
#           def train_test_small_dataset_alternate_model()

#
# Model 3:  def base_case_train_test()
#           def base_case_train_test_small_dataset()

#
# Model 4: def epoch_train_test()
#          def epoch_train_test_small_dataset()

#
# Model 5: def training_examples_train_test()
#          def training_examples_train_test_small_dataset()


""" Model 0: Using a fine-tuning BERT model with a dense layer of 2 units and a softmax actiavtion function
            MaxTrEg: 1000, MaxTeEg: 1000, Epoch: 3
            Using ac1Imdb dataset which contains 12,500 positive and 12,500 negative reviews
            Model uses training and testing examples from the provided dataset
            """
def train_test_intial_model() :
    """ Training and testing for text classification using BERT. """
    maxlen=512 # 512 maximum number of tokens
    maxTrEg=1000 # maximum number of pos & neg training examples
    maxTeEg=1000# maximum number of pos & neg test examples
    # read the data
    train_x, train_y = readPosNeg("aclImdb/train/pos","aclImdb/train/neg",maxTrEg)
    test_x, test_y = readPosNeg("aclImdb/test/pos","aclImdb/test/neg",maxTeEg)
    # tokenize train and test set
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_train = tokenizer(train_x, max_length=maxlen, truncation=True,
                                padding=True, return_tensors="tf")
    tokenized_test = tokenizer(test_x, max_length=maxlen, truncation=True,
                                padding=True, return_tensors="tf")
    # build the model
    bert_model = TFAutoModel.from_pretrained("distilbert-base-uncased")
    bert_model.trainable = False
    
    token_ids = Input(shape=(maxlen,), dtype=tf.int32,
                                      name="token_ids")
    attention_masks = Input(shape=(maxlen,), dtype=tf.int32,
                                            name="attention_masks")
    bert_output = bert_model(token_ids,attention_mask=attention_masks)
    output = Dense(2,activation="softmax")(bert_output[0][:,0])
    model = Model(inputs=[token_ids,attention_masks],outputs=output)
    # compile
    model.compile(optimizer="adam", loss="binary_crossentropy", 
metrics=["accuracy"])
    # train
    model.fit([tokenized_train["input_ids"],tokenized_train["attention_mask"]],
              train_y, batch_size=25, epochs=3)
    # evaluate
    score = model.evaluate([tokenized_test["input_ids"],tokenized_test["attention_mask"]],test_y,verbose=0)
    
    print("Accuracy on test data:",score[1])
    return model, [tokenized_train["input_ids"],tokenized_train["attention_mask"]],test_y


""" Model 0:  Using a fine-tuning BERT model with a dense layer of 2 units and a softmax actiavtion function
            MaxTrEg: 1000, MaxTeEg: 30, Epoch: 3
            Using small dataset which contains 15 positive and 15 negative reviews from popular movie review 
            webiste Imdb
            Model uses training and testing examples from the provided dataset"""

def train_test_small_dataset_intial_model() :
    """ Training and testing for text classification using BERT. """
    maxlen=512 # 512 maximum number of tokens
    maxTrEg=1000 # maximum number of pos & neg training examples
    maxTeEg=30# maximum number of pos & neg test examples
    # read the data
    train_x, train_y = readPosNeg("aclImdb/train/pos","aclImdb/train/neg",maxTrEg)
    test_x, test_y = readPosNeg("aclImdb/test/pos","aclImdb/test/neg",maxTeEg)
    # tokenize train and test set
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_train = tokenizer(train_x, max_length=maxlen, truncation=True,
                                padding=True, return_tensors="tf")
    tokenized_test = tokenizer(test_x, max_length=maxlen, truncation=True,
                                padding=True, return_tensors="tf")
    # build the model
    bert_model = TFAutoModel.from_pretrained("distilbert-base-uncased")
    bert_model.trainable = False
    
    token_ids = Input(shape=(maxlen,), dtype=tf.int32,
                                      name="token_ids")
    attention_masks = Input(shape=(maxlen,), dtype=tf.int32,
                                            name="attention_masks")
    bert_output = bert_model(token_ids,attention_mask=attention_masks)
    output = Dense(2,activation="softmax")(bert_output[0][:,0])
    model = Model(inputs=[token_ids,attention_masks],outputs=output)
    # compile
    model.compile(optimizer="adam", loss="binary_crossentropy", 
metrics=["accuracy"])
    # train
    model.fit([tokenized_train["input_ids"],tokenized_train["attention_mask"]],
              train_y, batch_size=25, epochs=3)
    # evaluate
    score = model.evaluate([tokenized_test["input_ids"],tokenized_test["attention_mask"]],test_y,verbose=0)
    
    print("Accuracy on test data:",score[1])
    return model, [tokenized_train["input_ids"],tokenized_train["attention_mask"]],test_y




""" Model 1: Using a fine-tuning BERT model with a dense layer of 2 units and a softmax actiavtion function
            MaxTrEg: 100, MaxTeEg: 100, Epoch: 6
            Using ac1Imdb dataset which contains 12,500 positive and 12,500 negative reviews
            Model uses training and testing examples from the provided dataset
            """
def train_test() :
    """ Training and testing for text classification using BERT. """
    maxlen=512  # 512 maximum number of tokens
    maxTrEg=100 # maximum number of pos & neg training examples
    maxTeEg=100 # maximum number of pos & neg test examples

    # read the data
    train_x, train_y = readPosNeg("aclImdb/train/pos","aclImdb/train/neg",maxTrEg)
    test_x, test_y = readPosNeg("aclImdb/test/pos","aclImdb/test/neg",maxTeEg)

    # tokenize train and test set
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_train = tokenizer(train_x, max_length=maxlen, truncation=True,
                                padding=True, return_tensors="tf")
    tokenized_test = tokenizer(test_x, max_length=maxlen, truncation=True,
                                padding=True, return_tensors="tf")

    # build the model
    bert_model = TFAutoModel.from_pretrained("distilbert-base-uncased")
    bert_model.trainable = False
    
    token_ids = Input(shape=(maxlen,), dtype=tf.int32,
                                      name="token_ids")
    attention_masks = Input(shape=(maxlen,), dtype=tf.int32,
                                            name="attention_masks")
    bert_output = bert_model(token_ids,attention_mask=attention_masks)

    output = Dense(2,activation="softmax")(bert_output[0][:,0])

    model = Model(inputs=[token_ids,attention_masks],outputs=output)

    # compile
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # train
    model.fit([tokenized_train["input_ids"],tokenized_train["attention_mask"]],
              train_y, batch_size=25, epochs=6)

    # evaluate
    score = model.evaluate([tokenized_test["input_ids"],tokenized_test["attention_mask"]],test_y,verbose=0)

    print("Loss on test data:", score[0])
    
    print("Accuracy on test data:",score[1])
    
    return model, [tokenized_train["input_ids"],tokenized_train["attention_mask"]], test_y


""" Model 1:  Using a fine-tuning BERT model with a dense layer of 2 units and a softmax actiavtion function
            MaxTrEg: 100, MaxTeEg: 30, Epoch: 6
            Using small dataset which contains 15 positive and 15 negative reviews from popular movie review 
            webiste Imdb
            Model uses training and testing examples from the provided dataset"""

def train_test_small_dataset() :
    # Evaluating for small dataset on fine-tuning model based on train_test()
    
    """ Training and testing for text classification using BERT. """
    maxlen=512  # 512 maximum number of tokens
    maxTrEg=100 # maximum number of pos & neg training examples
    maxTeEg=30  # maximum number of pos & neg test examples

    # read the data
    train_x, train_y = readPosNeg("aclImdb/train/pos","aclImdb/train/neg",maxTrEg)
    test_x, test_y = readPosNeg("small_dataset/test/pos_r","small_dataset/test/neg_r",maxTeEg)

    # tokenize train and test set
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_train = tokenizer(train_x, max_length=maxlen, truncation=True,
                                padding=True, return_tensors="tf")
    tokenized_test = tokenizer(test_x, max_length=maxlen, truncation=True,
                                padding=True, return_tensors="tf")

    # build the model
    bert_model = TFAutoModel.from_pretrained("distilbert-base-uncased")
    bert_model.trainable = False
    
    token_ids = Input(shape=(maxlen,), dtype=tf.int32,
                                      name="token_ids")
    attention_masks = Input(shape=(maxlen,), dtype=tf.int32,
                                            name="attention_masks")
    bert_output = bert_model(token_ids,attention_mask=attention_masks)

    output = Dense(2,activation="softmax")(bert_output[0][:,0])

    model = Model(inputs=[token_ids,attention_masks],outputs=output)

    # compile
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # train
    model.fit([tokenized_train["input_ids"],tokenized_train["attention_mask"]],
              train_y, batch_size=25, epochs=6)

    # evaluate
    score = model.evaluate([tokenized_test["input_ids"],tokenized_test["attention_mask"]],test_y,verbose=0)

    print("Loss on test data:", score[0])
    
    print("Accuracy on test data:",score[1])

    # predict
    a = model.predict([tokenized_test["input_ids"], tokenized_test["attention_mask"]])
    print(a.shape)
    for i in range(len(a)):
        print("Predicted=%s" % (a[i]))

    return model, [tokenized_train["input_ids"],tokenized_train["attention_mask"]], test_y



""" Model 2:  Using a fine-tuning BERT model with a dense layer of 2 units and a softmax actiavtion function with
            an additional dense layer with 64 units a with activation function ReLU, which takes in BERT’s output
            MaxTrEg: 100, MaxTeEg: 100, Epoch: 6
            Using ac1Imdb dataset which contains 12,500 positive and 12,500 negative reviews
            Model uses training and testing examples from the provided dataset """

def train_test_alternate_model() :
    # Evaluating for ac1Imdb dataset on alternate fine-tuning model differing in classification architecture
    
    """ Training and testing for text classification using BERT. """
    maxlen=512  # 512 maximum number of tokens
    maxTrEg=100 # maximum number of pos & neg training examples
    maxTeEg=100 # maximum number of pos & neg test examples

    # read the data
    train_x, train_y = readPosNeg("aclImdb/train/pos","aclImdb/train/neg",maxTrEg)
    test_x, test_y = readPosNeg("aclImdb/test/pos","aclImdb/test/neg",maxTeEg)

    # tokenize train and test set
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_train = tokenizer(train_x, max_length=maxlen, truncation=True,
                                padding=True, return_tensors="tf")
    tokenized_test = tokenizer(test_x, max_length=maxlen, truncation=True,
                                padding=True, return_tensors="tf")

    # build the model
    bert_model = TFAutoModel.from_pretrained("distilbert-base-uncased")
    bert_model.trainable = False
    
    token_ids = Input(shape=(maxlen,), dtype=tf.int32,name="token_ids")
    attention_masks = Input(shape=(maxlen,), dtype=tf.int32,name="attention_masks")
    bert_output = bert_model(token_ids,attention_mask=attention_masks)

    a = Dense(64,activation="relu")(bert_output[0][:,0])
    output = Dense(2, activation="softmax")(a)

    model = Model(inputs=[token_ids,attention_masks],outputs=output)

    # compile
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # train
    model.fit([tokenized_train["input_ids"],tokenized_train["attention_mask"]],
              train_y, batch_size=25, epochs=6)

    # evaluate
    score = model.evaluate([tokenized_test["input_ids"],tokenized_test["attention_mask"]],test_y,verbose=0)

    print("Loss on test data:", score[0])
    
    print("Accuracy on test data:",score[1])
    
    return model, [tokenized_train["input_ids"],tokenized_train["attention_mask"]], test_y


""" Model 2:  Using a fine-tuning BERT model with a dense layer of 2 units and a softmax actiavtion function with
            an additional dense layer with 64 units a with activation function ReLU, which takes in BERT’s output
            MaxTrEg: 100, MaxTeEg: 100, Epoch: 6
            Using small dataset which contains 15 positive and 15 negative reviews from popular movie review 
            webiste Imdb
            Model uses training and testing examples from the provided dataset """

def train_test_small_dataset_alternate_model() :
    # Evaluating for small dataset on alternate fine-tuning model differing in classification architecture
    
    """ Training and testing for text classification using BERT. """
    maxlen=512  # 512 maximum number of tokens
    maxTrEg=100 # maximum number of pos & neg training examples
    maxTeEg=30  # maximum number of pos & neg test examples

    # read the data
    train_x, train_y = readPosNeg("aclImdb/train/pos","aclImdb/train/neg",maxTrEg)
    test_x, test_y = readPosNeg("small_dataset/test/pos_r","small_dataset/test/neg_r",maxTeEg)

    # tokenize train and test set
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_train = tokenizer(train_x, max_length=maxlen, truncation=True,
                                padding=True, return_tensors="tf")
    tokenized_test = tokenizer(test_x, max_length=maxlen, truncation=True,
                                padding=True, return_tensors="tf")

    # build the model
    bert_model = TFAutoModel.from_pretrained("distilbert-base-uncased")
    bert_model.trainable = False
    
    token_ids = Input(shape=(maxlen,), dtype=tf.int32,name="token_ids")
    attention_masks = Input(shape=(maxlen,), dtype=tf.int32,name="attention_masks")
    bert_output = bert_model(token_ids,attention_mask=attention_masks)

    a = Dense(64,activation="relu")(bert_output[0][:,0])
    output = Dense(2, activation="softmax")(a)

    model = Model(inputs=[token_ids,attention_masks],outputs=output)

    # compile
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # train
    model.fit([tokenized_train["input_ids"],tokenized_train["attention_mask"]],
              train_y, batch_size=25, epochs=6)

    # evaluate
    score = model.evaluate([tokenized_test["input_ids"],tokenized_test["attention_mask"]],test_y,verbose=0)

    print("Loss on test data:", score[0])
    
    print("Accuracy on test data:",score[1])

    # predict
    a = model.predict([tokenized_test["input_ids"], tokenized_test["attention_mask"]])
    print(a.shape)
    for i in range(len(a)):
        print("Predicted=%s" % (a[i]))

    return model, [tokenized_train["input_ids"],tokenized_train["attention_mask"]], test_y



""" Model 3: Similar to Model 1, differs in distilbert's base-cased being used for this case
            MaxTrEg: 100, MaxTeEg: 100, Epoch: 6
            Using ac1Imdb dataset which contains 12,500 positive and 12,500 negative reviews
            Model uses training and testing examples from the provided dataset """


def base_case_train_test() :
    # Evaluating for ac1Imdb dataset for base-cased
    
    """ Training and testing for text classification using BERT. """
    maxlen=512  # 512 maximum number of tokens
    maxTrEg=100 # maximum number of pos & neg training examples
    maxTeEg=100 # maximum number of pos & neg test examples

    # read the data
    train_x, train_y = readPosNeg("aclImdb/train/pos","aclImdb/train/neg",maxTrEg)
    test_x, test_y = readPosNeg("aclImdb/test/pos","aclImdb/test/neg",maxTeEg)

    # tokenize train and test set
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    tokenized_train = tokenizer(train_x, max_length=maxlen, truncation=True,
                                padding=True, return_tensors="tf")
    tokenized_test = tokenizer(test_x, max_length=maxlen, truncation=True,
                                padding=True, return_tensors="tf")

    # build the model
    bert_model = TFAutoModel.from_pretrained("distilbert-base-cased")
    bert_model.trainable = False
    
    token_ids = Input(shape=(maxlen,), dtype=tf.int32,
                                      name="token_ids")
    attention_masks = Input(shape=(maxlen,), dtype=tf.int32,
                                            name="attention_masks")
    bert_output = bert_model(token_ids,attention_mask=attention_masks)

    output = Dense(2,activation="softmax")(bert_output[0][:,0])

    model = Model(inputs=[token_ids,attention_masks],outputs=output)

    # compile
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # train
    model.fit([tokenized_train["input_ids"],tokenized_train["attention_mask"]],
              train_y, batch_size=25, epochs=6)

    # evaluate
    score = model.evaluate([tokenized_test["input_ids"],tokenized_test["attention_mask"]],test_y,verbose=0)

    print("Loss on test data:", score[0])
    
    print("Accuracy on test data:",score[1])

    return model, [tokenized_train["input_ids"],tokenized_train["attention_mask"]], test_y


""" Model 3:  Similar to Model 1, differs in distilbert's base-cased being used for this case
            MaxTrEg: 100, MaxTeEg: 30, Epoch: 6
            Using small dataset which contains 15 positive and 15 negative reviews from popular movie review 
            webiste Imdb
            Model uses training and testing examples from the provided dataset"""

def base_case_train_test_small_dataset() :
    # Evaluating for small dataset for base-cased
    
    """ Training and testing for text classification using BERT. """
    maxlen=512  # 512 maximum number of tokens
    maxTrEg=100 # maximum number of pos & neg training examples
    maxTeEg=30  # maximum number of pos & neg test examples

    # read the data
    train_x, train_y = readPosNeg("aclImdb/train/pos","aclImdb/train/neg",maxTrEg)
    test_x, test_y = readPosNeg("small_dataset/test/pos_r","small_dataset/test/neg_r",maxTeEg)

    # tokenize train and test set
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    tokenized_train = tokenizer(train_x, max_length=maxlen, truncation=True,
                                padding=True, return_tensors="tf")
    tokenized_test = tokenizer(test_x, max_length=maxlen, truncation=True,
                                padding=True, return_tensors="tf")

    # build the model
    bert_model = TFAutoModel.from_pretrained("distilbert-base-cased")
    bert_model.trainable = False
    
    token_ids = Input(shape=(maxlen,), dtype=tf.int32,
                                      name="token_ids")
    attention_masks = Input(shape=(maxlen,), dtype=tf.int32,
                                            name="attention_masks")
    bert_output = bert_model(token_ids,attention_mask=attention_masks)

    output = Dense(2,activation="softmax")(bert_output[0][:,0])

    model = Model(inputs=[token_ids,attention_masks],outputs=output)

    # compile
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # train
    model.fit([tokenized_train["input_ids"],tokenized_train["attention_mask"]],
              train_y, batch_size=25, epochs=6)

    # evaluate
    score = model.evaluate([tokenized_test["input_ids"],tokenized_test["attention_mask"]],test_y,verbose=0)

    print("Loss on test data:", score[0])
    
    print("Accuracy on test data:",score[1])

    # predict
    a = model.predict([tokenized_test["input_ids"], tokenized_test["attention_mask"]])
    print(a.shape)
    for i in range(len(a)):
        print("Predicted=%s" % (a[i]))

    return model, [tokenized_train["input_ids"],tokenized_train["attention_mask"]], test_y
    


""" Model 4: Similar to Model 1, differs in number of epochs used, this model can be used 
            to see how accuracy differs
            MaxTrEg: 100, MaxTeEg: 100, Epoch: 15
            Using ac1Imdb dataset which contains 12,500 positive and 12,500 negative reviews
            Model uses training and testing examples from the provided dataset """

def epoch_train_test() :
    # Evaluating for ac1Imdb dataset on greater number of epochs

    """ Training and testing for text classification using BERT. """
    maxlen=512  # 512 maximum number of tokens
    maxTrEg=100 # maximum number of pos & neg training examples
    maxTeEg=100 # maximum number of pos & neg test examples

    # read the data
    train_x, train_y = readPosNeg("aclImdb/train/pos","aclImdb/train/neg",maxTrEg)
    test_x, test_y = readPosNeg("aclImdb/test/pos","aclImdb/test/neg",maxTeEg)

    # tokenize train and test set
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_train = tokenizer(train_x, max_length=maxlen, truncation=True,
                                padding=True, return_tensors="tf")
    tokenized_test = tokenizer(test_x, max_length=maxlen, truncation=True,
                                padding=True, return_tensors="tf")

    # build the model
    bert_model = TFAutoModel.from_pretrained("distilbert-base-uncased")
    bert_model.trainable = False
    
    token_ids = Input(shape=(maxlen,), dtype=tf.int32,
                                      name="token_ids")
    attention_masks = Input(shape=(maxlen,), dtype=tf.int32,
                                            name="attention_masks")
    bert_output = bert_model(token_ids,attention_mask=attention_masks)

    output = Dense(2,activation="softmax")(bert_output[0][:,0])

    model = Model(inputs=[token_ids,attention_masks],outputs=output)

    # compile
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # train
    model.fit([tokenized_train["input_ids"],tokenized_train["attention_mask"]],
              train_y, batch_size=25, epochs=15)

    # evaluate
    score = model.evaluate([tokenized_test["input_ids"],tokenized_test["attention_mask"]],test_y,verbose=0)

    print("Loss on test data:", score[0])
    
    print("Accuracy on test data:",score[1])
    
    return model, [tokenized_train["input_ids"],tokenized_train["attention_mask"]], test_y


""" Model 4: Similar to Model 1, differs in number of epochs used, this model can be used 
            to see how accuracy differs
            MaxTrEg: 100, MaxTeEg: 30, Epoch: 15
            Using small dataset which contains 15 positive and 15 negative reviews from popular movie review 
            webiste Imdb
            Model uses training and testing examples from the provided dataset """

def epoch_train_test_small_dataset() :
    # Evaluating for small dataset on greater number of epochs

    """ Training and testing for text classification using BERT. """
    maxlen=512  # 512 maximum number of tokens
    maxTrEg=100 # maximum number of pos & neg training examples
    maxTeEg=30  # maximum number of pos & neg test examples

    # read the data
    train_x, train_y = readPosNeg("aclImdb/train/pos","aclImdb/train/neg",maxTrEg)
    test_x, test_y = readPosNeg("small_dataset/test/pos_r","small_dataset/test/neg_r",maxTeEg)

    # tokenize train and test set
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_train = tokenizer(train_x, max_length=maxlen, truncation=True,
                                padding=True, return_tensors="tf")
    tokenized_test = tokenizer(test_x, max_length=maxlen, truncation=True,
                                padding=True, return_tensors="tf")

    # build the model
    bert_model = TFAutoModel.from_pretrained("distilbert-base-uncased")
    bert_model.trainable = False
    
    token_ids = Input(shape=(maxlen,), dtype=tf.int32,
                                      name="token_ids")
    attention_masks = Input(shape=(maxlen,), dtype=tf.int32,
                                            name="attention_masks")
    bert_output = bert_model(token_ids,attention_mask=attention_masks)

    output = Dense(2,activation="softmax")(bert_output[0][:,0])

    model = Model(inputs=[token_ids,attention_masks],outputs=output)

    # compile
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # train
    model.fit([tokenized_train["input_ids"],tokenized_train["attention_mask"]],
              train_y, batch_size=25, epochs=15)

    # evaluate
    score = model.evaluate([tokenized_test["input_ids"],tokenized_test["attention_mask"]],test_y,verbose=0)

    print("Loss on test data:", score[0])
    
    print("Accuracy on test data:",score[1])

    # predict
    a = model.predict([tokenized_test["input_ids"], tokenized_test["attention_mask"]])
    print(a.shape)
    for i in range(len(a)):
        print("Predicted=%s" % (a[i]))

    return model, [tokenized_train["input_ids"],tokenized_train["attention_mask"]], test_y




""" Model 5: Similar to Model 1, differs in number of training examples used
            MaxTrEg: 250, MaxTeEg: 100, Epoch: 6
            Using ac1Imdb dataset which contains 12,500 positive and 12,500 negative reviews
            Model uses training and testing examples from the provided dataset """

def training_examples_train_test() :
    # Evaluating for ac1Imdb dataset on greater number of training examples
    
    """ Training and testing for text classification using BERT. """
    maxlen=512  # 512 maximum number of tokens
    maxTrEg=250 # maximum number of pos & neg training examples
    maxTeEg=100 # maximum number of pos & neg test examples

    # read the data
    train_x, train_y = readPosNeg("aclImdb/train/pos","aclImdb/train/neg",maxTrEg)
    test_x, test_y = readPosNeg("aclImdb/test/pos","aclImdb/test/neg",maxTeEg)

    # tokenize train and test set
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_train = tokenizer(train_x, max_length=maxlen, truncation=True,
                                padding=True, return_tensors="tf")
    tokenized_test = tokenizer(test_x, max_length=maxlen, truncation=True,
                                padding=True, return_tensors="tf")

    # build the model
    bert_model = TFAutoModel.from_pretrained("distilbert-base-uncased")
    bert_model.trainable = False
    
    token_ids = Input(shape=(maxlen,), dtype=tf.int32,
                                      name="token_ids")
    attention_masks = Input(shape=(maxlen,), dtype=tf.int32,
                                            name="attention_masks")
    bert_output = bert_model(token_ids,attention_mask=attention_masks)

    output = Dense(2,activation="softmax")(bert_output[0][:,0])

    model = Model(inputs=[token_ids,attention_masks],outputs=output)

    # compile
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # train
    model.fit([tokenized_train["input_ids"],tokenized_train["attention_mask"]],
              train_y, batch_size=25, epochs=6)

    # evaluate
    score = model.evaluate([tokenized_test["input_ids"],tokenized_test["attention_mask"]],test_y,verbose=0)

    print("Loss on test data:", score[0])
    
    print("Accuracy on test data:",score[1])
    
    return model, [tokenized_train["input_ids"],tokenized_train["attention_mask"]], test_y


""" Model 5: Similar to Model 1, differs in number of training examples
            MaxTrEg: 250, MaxTeEg: 30, Epoch: 6
            Using small dataset which contains 15 positive and 15 negative reviews from popular movie review 
            webiste Imdb
            Model uses training and testing examples from the provided dataset """

def training_examples_train_test_small_dataset() :
    # Evaluating for ac1Imdb dataset on greater number of training examples
    
    """ Training and testing for text classification using BERT. """
    maxlen=512  # 512 maximum number of tokens
    maxTrEg=250 # maximum number of pos & neg training examples
    maxTeEg=30  # maximum number of pos & neg test examples

    # read the data
    train_x, train_y = readPosNeg("aclImdb/train/pos","aclImdb/train/neg",maxTrEg)
    test_x, test_y = readPosNeg("small_dataset/test/pos_r","small_dataset/test/neg_r",maxTeEg)

    # tokenize train and test set
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_train = tokenizer(train_x, max_length=maxlen, truncation=True,
                                padding=True, return_tensors="tf")
    tokenized_test = tokenizer(test_x, max_length=maxlen, truncation=True,
                                padding=True, return_tensors="tf")

    # build the model
    bert_model = TFAutoModel.from_pretrained("distilbert-base-uncased")
    bert_model.trainable = False
    
    token_ids = Input(shape=(maxlen,), dtype=tf.int32,
                                      name="token_ids")
    attention_masks = Input(shape=(maxlen,), dtype=tf.int32,
                                            name="attention_masks")
    bert_output = bert_model(token_ids,attention_mask=attention_masks)

    output = Dense(2,activation="softmax")(bert_output[0][:,0])

    model = Model(inputs=[token_ids,attention_masks],outputs=output)

    # compile
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # train
    model.fit([tokenized_train["input_ids"],tokenized_train["attention_mask"]],
              train_y, batch_size=25, epochs=6)

    # evaluate
    score = model.evaluate([tokenized_test["input_ids"],tokenized_test["attention_mask"]],test_y,verbose=0)

    print("Loss on test data:", score[0])
    
    print("Accuracy on test data:",score[1])

    # predict
    a = model.predict([tokenized_test["input_ids"], tokenized_test["attention_mask"]])
    print(a.shape)
    for i in range(len(a)):
        print("Predicted=%s" % (a[i]))
    
    return model, [tokenized_train["input_ids"],tokenized_train["attention_mask"]], test_y
