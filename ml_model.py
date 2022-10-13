# 1) Design model (input size, output size, forward pass)
# 2) Construct loss and optimizer
#   - loss determines difference between calculated label and actual label
#   - optimizer works to update weights to minimize loss
# 3) Training loop:
#   - forward pass: compute prediction and loss
#   - backward pass: compute gradients
#   - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) prepare data

# Use sklearn breast cancer dataset at https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
# This dataset contains a binary classification (either benign or malignant) and has several inputs describing different cell characteristics
# X - holds data with shape (569,30), meaning there are 30 inputs (features) and 569 different pieces of data (samples)
# y - holds labels with shape (569,)
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape
print(n_samples, n_features)

# Below command splits your current data into a training set and a testing set
# test_size - percentage to set as test data
# random_state - int number specifying how much to randomize data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale
sc = StandardScaler()

# Use fit_transform on training data and transform on testing data
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Convert to torch tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# Reshape vectors using torch.view()
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


# 1) model
# f = wx + b, sigmoid function applied at end
class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear1 = nn.Linear(n_input_features, 16)
        self.linear2 = nn.Linear(16, 8)
        self.linear3 = nn.Linear(8, 1)

        #self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted1 = self.linear1(x)
        y_predicted2 = self.linear2(y_predicted1)
        y_predicted3 = self.linear3(y_predicted2)
        y_predicted = torch.sigmoid(y_predicted3)

        #y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features)

# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss calculation
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    # backward pass
    loss.backward()

    # updates
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# 4) testing set
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')