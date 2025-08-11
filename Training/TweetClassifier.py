import mlflow
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from keras.layers import TextVectorization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential # from tf_keras import Sequential
from keras.layers import Dense, Embedding, Flatten, LSTM
from keras.saving import register_keras_serializable
import tensorflow as tf
from .Preproc import tokenize_and_preprocess

@register_keras_serializable()
class KerasTweetClassifier(Sequential):

    def __init__(self, max_length=128,
                    top_words = 10000,
                    type = 'SIMPLE',
                    vocabulary =  None,
                    loss = 'categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'],
                    include_vectorizer = True,
                    **kwarg,
                    ):
        super().__init__()
    
        embedding_vector_length = 32

        if include_vectorizer:
            self.add(TextVectorization(max_tokens=top_words,
                                        standardize=None,
                                        split='whitespace',
                                        ngrams=None,
                                        output_mode='int',
                                        output_sequence_length = max_length,
                                        vocabulary=vocabulary,
                                        ))

        self.add(Embedding(top_words, embedding_vector_length, input_length=max_length))
        
        match type:
            case 'SIMPLE':
                self.add(Flatten())
                self.add(Dense(16, activation='relu'))
                self.add(Dense(16, activation='relu'))
                self.add(Dense(1, activation='sigmoid'))
            case 'LTSM':
                lstm_out = 200
                self.add(LSTM(lstm_out, dropout_U = 0.2, dropout_W = 0.2))
                self.add(Dense(2,activation='softmax'))

        self.compile(loss=loss,optimizer=optimizer, metrics=metrics)
    
    def get_config(self):
        config = super(KerasTweetClassifier, self).get_config()
        base_config = {
            'max_length': 128,
            'top_words': 10000,
            'type': 'SIMPLE',
            'vocabulary': None,
            'loss': 'categorical_crossentropy',
            'optimizer': 'adam',
            'metrics': ['accuracy'],
            'include_vectorizer': True
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls( **config)


    # def set_vocabulary(self, texts):
    #     if self.include_vectorizer:
    #         self.layers[0].adapt(texts)

    # def get_config(self):
    #     return {"factor": self.factor}


class TweetClassifierPipeline(mlflow.pyfunc.PythonModel,
                            #   BaseEstimator, 
                            #   TransformerMixin
                              ):
    def __init__(self, preproc_actions=[],
                 preproc_func = tokenize_and_preprocess,
                 vector_lib = 'SKLEARN',
                 vectorizer_func=CountVectorizer, 
                 vectorizer_params={}, 
                 model_lib = 'SKLEARN',
                 model_func=LogisticRegression, 
                 model_params={}):
        self.preproc_func = preproc_func
        self.preproc_actions = preproc_actions
        self.vectorizer_func = vectorizer_func
        self.vectorizer_params = vectorizer_params
        self.model_lib = model_lib
        self.model_func = model_func
        self.model_params = model_params
        self.vector_lib = vector_lib


    def fit(self, X, y=None,x_test=None,y_test=None):

        if len(self.preproc_actions) > 0:
            X = self.preproc(X)
            if type(x_test)!=type(None):
                x_test = self.preproc(x_test)


        if type(self.vectorizer_func)!=type(None):
            self.vectorizer_fit(X)
            X =  self.vectorize_transform(X)
        


        match self.model_lib:
            case "KERAS":
                if not self.model_params.get('vocabulary'):
                    '''Need to adapt vector layer vocabulary too'''
                    tokenized,word_frequ_df = self.preproc_func(X,                              
                                actions=['TOKENISE',
                                         'WORD_FREQ'],
                                return_frequ=True,
                                )
                    max_words = self.model_params.get('top_words',500)
                    
                    vocabulary = word_frequ_df['Word'].head(max_words).tolist()
                    self.model_params['vocabulary']=vocabulary
                model = self.model_func(**self.model_params) 
                # X = X.tolist()
                # x_test = x_test.tolist()


                # X = [["foo qux bar"], ["qux baz"]]
                # x_test = [["foo qux bar"], ["qux baz"]]
                # y= [0,1]
                # y_test= [0,1]

                # 
                
                X = np.asarray(X).astype('str')
                X = tf.convert_to_tensor(X) 
                # X = tf.constant(X)
                y = np.array(y).astype(int)
                
                x_test = np.asarray(x_test).astype('str')
                x_test = tf.convert_to_tensor(x_test) 
                x_test = tf.constant(x_test)
                y_test = np.array(y_test).astype(int)

                es = EarlyStopping(monitor=self.model_params.get('monitor','val_loss'),
                        mode='min', 
                        verbose=1, 
                        patience=self.model_params.get('patience',5))
        
                callbacks_list = [es]
                
                model.fit(X, y, validation_data=(x_test, y_test),
                           epochs=self.model_params.get('epoch',5),
                           batch_size=self.model_params.get('batch_size',128),
                           callbacks=callbacks_list,                            
                        )

            case _:

                model = self.model_func(**self.model_params)
                model.fit(X, y)

        # self.vectorizer_model_ = self.vectorizer_model
        self.model_ = model

        return self
    
    def vectorizer_fit(self,X):
        match self.vector_lib:
            case 'SKLEARN':
                self.vectorizer_model =  self.vectorizer_func(**self.vectorizer_params)
                # X = vectorizer_model.fit_transform(X)
                self.vectorizer_model.fit(X)
                self.vectorize_transform = self.vectorizer_model.transform

            case 'GENSIM':
                self.vectorizer_model =  self.vectorizer_func(sentences = X,
                                                    **self.vectorizer_params
                                                    )
                def gensim_transformer(X):
                    def get_sentence_vector(tokens_list):
                        vectors = [self.vectorizer_model.wv[token] for token in tokens_list if token in self.vectorizer_model.wv]
                        if len(vectors) == 0:
                            return np.zeros(self.vectorizer_model.vector_size)
                        return np.mean(vectors, axis=0)

                    return np.stack(X.apply(get_sentence_vector))
                self.vectorize_transform = gensim_transformer
                
            case 'SENTENCE_TRANSFORMERS':
                self.vectorizer_model =  self.vectorizer_func(**self.vectorizer_params)
                def st_transformer(X):
                    return self.vectorizer_model.encode(X.to_list(),
                                                show_progress_bar=True)
                self.vectorize_transform = st_transformer

                

            case _:
                self.vectorizer_model.fit(X)
                self.vectorize_transform = self.vectorizer_model.transform
        

    def predict(self, X, make_binary=True):
        if len(self.preproc_actions) > 0:
            X = self.preproc(X)

        if type(self.vectorizer_func)!=type(None):
            X = self.vectorize_transform(X)
        
        if self.model_lib == 'KERAS':
            X = tf.constant(X)

        prediction = self.model_.predict(X)

        if make_binary:            
            return np.where(prediction > 0.5, 1, 0)

        return prediction

    def preproc(self,X):        
        return self.preproc_func(X,                              
                            actions=self.preproc_actions,
                            )
        

    def predict_proba(self, X):
        return self.predict(X,make_binary=False)