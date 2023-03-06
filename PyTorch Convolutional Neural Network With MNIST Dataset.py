import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

train_data = datasets.MNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True
)

test_data = datasets.MNIST(
    root='data',
    train=False,
    transform=ToTensor(),
    download=True
)


# print(train_data.data.size())
# print(train_data.targets.size())

# Plot one train_data
# plt.imshow(train_data.data[0], cmap='gray')
# plt.title('%i' % train_data.targets[0])
# plt.show()
#
# # Plot multiple train_data
# figure = plt.figure(figsize=(10,8))
# cols, rows = 5, 5
# for i in range(1, cols*rows+1):
#     sample_idx = torch.randint(len(train_data), size=(1,)).item()
#     img, label  = train_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(label)
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()

# Define the Convolutional Neural Network model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x  # return x for visualization


def train(num_epochs, cnn, loaders, optimizer, loss_func):
    cnn.train()

    # Train the model
    total_step = len(loaders['train'])

    for epoch in range(num_epochs):
        epoch_loss = 0
        batch_count = 0

        average_batch_loss = 0
        for i, (images, labels) in enumerate(loaders['train']):
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)  # batch x
            b_y = Variable(labels)  # batch y

            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)

            # clear gradients for this training step
            optimizer.zero_grad()

            # backpropagation, compute gradients
            loss.backward()
            # apply gradients
            optimizer.step()

            average_batch_loss += loss.item()
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, average_batch_loss/100))
                average_batch_loss = 0

            epoch_loss += loss.item()
            batch_count += 1

        print(f"Epoch loss: {epoch_loss/batch_count}")


# Evaluate the model on test data
def test(cnn, loaders):
    # Test the model
    cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            correct += (pred_y == labels).sum().item()
            total += labels.size(0)
        accuracy = correct / total

        print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)


if __name__ == '__main__':
    loaders = {
        'train': torch.utils.data.DataLoader(train_data,
                                             batch_size=100,
                                             shuffle=True,
                                             num_workers=1),

        'test': torch.utils.data.DataLoader(test_data,
                                            batch_size=100,
                                            shuffle=True,
                                            num_workers=1),
    }

    cnn = CNN()
    print(cnn)
    # Define loss function
    loss_func = nn.CrossEntropyLoss()
    # Define a Optimization Function
    optimizer = optim.Adam(cnn.parameters(), lr=0.001)
    # Train the model
    num_epochs = 5

    train(num_epochs, cnn, loaders, optimizer, loss_func)

    test(cnn, loaders)


