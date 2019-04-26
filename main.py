import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as mnistdt
from Model.LSTMmodel import LSTMModel


# setup the trainning and test data

traindt = mnistdt.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

testdt = mnistdt.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

print('Trainning dataset size', traindt.train_data.size())
print('Trainning dataset labels', traindt.train_labels.size())
print('Testing dataset size', testdt.test_data.size())
print('Testning dataset labels', testdt.test_labels.size())

batch_size = 100
n_iters = 3000

num_epochs = n_iters / (len(traindt) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=traindt,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=testdt,
                                          batch_size=batch_size,
                                          shuffle=False)

input_dim = 28
hidden_dim = 100
layer_dim = 1
output_dim = 10

model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)

# define loss function and setup optimizer
criterion = nn.CrossEntropyLoss()

learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())

# Number of steps to unroll
seq_dim = 28

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        images = images.view(-1, seq_dim, input_dim).requires_grad_()

        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)

        # calculate Loss
        loss = criterion(outputs, labels)

        # apply backpropagate
        loss.backward()

        # update
        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                images = images.view(-1, seq_dim, input_dim).requires_grad_()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)

                #  correct predictions
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / total

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))

