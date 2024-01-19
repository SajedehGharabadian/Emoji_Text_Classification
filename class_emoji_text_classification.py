import tensorflow as tf
import numpy as np
import pandas as pd
import time
import argparse


class EmojiTextClassifier:
    def __init__(self,vector_shape):
        self.vector_shape = vector_shape

    def load_dataset(self,dataset_path):
        self.df = pd.read_csv(dataset_path)
        self.X = np.array(self.df['sentence'],dtype=object)
        self.Y = np.array(self.df['label'],dtype=int)
        return self.X,self.Y

    def load_features_vector(self,file_txt_path):
        self.f = open(file_txt_path,encoding='utf-8')
        self.word_vectors = {}
        for line in self.f:
            line = line.strip().split()
            word = line[0]
            vector = np.array(line[1:],dtype=np.float64)
            self.word_vectors[word] = vector

        return self.word_vectors


    def sentence_to_feature_vectors_avg(self,sentence):
          self.sentence = sentence.lower()
          words = self.sentence.strip().split(' ')
          sum_vectors = np.zeros((self.vector_shape,))
          for word in words:
              sum_vectors += self.word_vectors[word]

          avg_words = sum_vectors / len(words)

          return avg_words


    def preprocess(self,X,Y):
        self.X_avg = []
        self.X = X
        self.Y = Y
        for x in X:
            self.X_avg.append(self.sentence_to_feature_vectors_avg(x))


        self.X_avg = np.array(self.X_avg).astype('float32')
        self.Y_one_hot = tf.keras.utils.to_categorical(self.Y,num_classes=5)

        return self.X_avg,self.Y_one_hot

    def load_model(self):

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(5,input_shape=(self.vector_shape,),activation='softmax')
        ])
        return model

    def train(self,X_train,Y_train,epochs):
        #self.input_shape =input_shape
        self.X_train_avg,self.Y_train_one_hot = self.preprocess(X_train,Y_train)
        print(self.X_train_avg.shape,self.Y_train_one_hot.shape)
        self.model = self.load_model()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        #self.model.fit(self.X_train_avg,self.Y_train_one_hot,epochs)
        return self.model

    def test(self,model,X_test,Y_test):
        self.X_test = X_test
        self.Y_test = Y_test
        #self.X_test_avg,self.Y_test_one_hot = self.preprocess(self.X_test,self.Y_test)
        self.X_avg = []
        self.new_Y_test = []
        i = 0

        for index,x in enumerate(self.X_test):
            if x.endswith('\t') == True:
              self.X_avg.append(self.sentence_to_feature_vectors_avg(x))
              self.new_Y_test.append(self.Y_test[index])



        self.X_avg = np.array(self.X_avg)
        self.new_Y_test = np.array(self.new_Y_test)
        self.Y_one_hot = tf.keras.utils.to_categorical(self.new_Y_test,num_classes=5)
        print(np.shape(self.X_avg),np.shape(self.Y_one_hot))

        accuracy,loss = model.evaluate(self.X_avg,self.Y_one_hot)

        return accuracy,loss

    def label_to_emoji(self,label):
        self.label = label
        emojies=['❤️','⚾','😊','😞','🍴']

        return emojies[self.label]

    def predict(self,sentece_test):
        start_time = time.time()
        self.sentence_test = sentece_test
        self.my_test_avg = self.sentence_to_feature_vectors_avg(self.sentence_test)
        self.my_test_avg = np.array([self.my_test_avg])
        self.result = self.model.predict(self.my_test_avg)
        y_pred = np.argmax(self.result)

        return self.label_to_emoji(y_pred),time.time() - start_time


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--vector_shape',type=int)
    parser.add_argument('--features_path',type=str,help='path')
    parser.add_argument('--inference',type=str)

    opt = parser.parse_args()
    
    emoji_text = EmojiTextClassifier(opt.vector_shape)
    X_train,Y_train = emoji_text.load_dataset('/content/drive/MyDrive/Emoji_Text_Classification/train.csv')
    X_test,Y_test = emoji_text.load_dataset('/content/drive/MyDrive/Emoji_Text_Classification/test.csv')
    emoji_text.load_features_vector(opt.features_path)
    X_train_avg,Y_train_one_hot = emoji_text.preprocess(X_train,Y_train)
    model = emoji_text.train(X_train,Y_train,epochs=300)
    model.fit(X_train_avg,Y_train_one_hot,epochs=300)
    loss,accuracy= emoji_text.test(model,X_test,Y_test)
    print(loss,accuracy)
    emoji,timer = emoji_text.predict(opt.inference)
    print(emoji)




