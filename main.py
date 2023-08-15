import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier

data = fetch_openml("mnist_784", version=1)
print(data)

x, y = data["data"], data["target"]
print(x.shape)

# split the data into training and test
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# sample of images in the dataset
image = np.array(xtrain.iloc[0]).reshape(28, 28)
plt.imshow(image)

model = SGDClassifier()
model.fit(xtrain, ytrain)

# test the trained model by making predictions on the test set
predictions = model.predict(xtest)
print(predictions)

# lets look at the digit images
image = np.array(xtest.iloc[0]).reshape(28, 28)
plt.imshow(image)

image = np.array(xtest.iloc[1]).reshape(28, 28)
plt.imshow(image)




