import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, cross_validation, metrics
from sklearn import preprocessing
from tensorflow.contrib import learn
from keras.models import Sequential
from keras.layers import Dense

# Read the original dataset
df = pd.read_csv("C:/Users/cuong_000/PycharmProjects/GoogleFoodAnalysis/mpg.csv", header=0)
# Convert the displacement column as float
df['displacement']=df['displacement'].astype(float)
# We get data columns from the dataset
# First and last (mpg and car names) are ignored for X
X = df[df.columns[1:8]]
y = df['mpg']

plt.figure() # Create a new figure
f, ax1 = plt.subplots()
for i in range (1,8):
    number = 420 + i
    ax1.locator_params(nbins=3)
    ax1 = plt.subplot(number)
    plt.title(list(df)[i])
    ax1.scatter(df[df.columns[i]],y) #Plot a scatter draw of the  datapoints
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# Split the datasets

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
    test_size=0.25)

# Scale the data for convergency optimization
scaler = preprocessing.StandardScaler()

# Set the transform parameters
X_train = scaler.fit_transform(X_train)

# Build a 2 layer fully connected DNN with 10 and 5 units respectively

model = Sequential()
model.add(Dense(10, input_dim=7, init='normal', activation='relu'))
model.add(Dense(5, init='normal', activation='relu'))
model.add(Dense(1, init='normal'))

#Compile the model, whith the mean squared error as a loss function
model.compile(loss='mean_squared_error', optimizer='adam')

#Fit the model, in 1000 epochs
model.fit(X_train, y_train, nb_epoch=1000, validation_split=0.33, shuffle=True,verbose=2 )