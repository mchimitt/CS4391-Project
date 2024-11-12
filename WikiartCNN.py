import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# Here we define our CNN archintecture for the different genres
class GenreClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super(GenreClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)#1st conv layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)#2nd conv layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)#3rd con layer
        self.fc1 = nn.Linear(128 * 16 * 16, 512)  # Adjust based on image size after conv layers
        self.fc2 = nn.Linear(512, num_classes)  # Output layer for genre classes they also connect the layers


        #this is always called by pytorch automatically
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))#max pooling the 1st also introduces ono linear
        x = F.relu(F.max_pool2d(self.conv2(x), 2))#max pools the 2nd
        x = F.relu(F.max_pool2d(self.conv3(x), 2))# max pools the 3rd
        x = x.view(-1, 128 * 16 * 16)  # Flattens the final output
        x = F.relu(self.fc1(x))#
        x = self.fc2(x)
        return x

# Function to load dataset and return a DataLoader
def get_data_loader(data_dir, batch_size=32):
    #resizes the image to 128x 128
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        #converts image to pytorch sensor
        transforms.ToTensor(),
        #normalizes pixels
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    #loads the data from directory
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    #create dataloader which shuffles around
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader, dataset.classes

# Training function modes
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    #loop through epoch
    for epoch in range(num_epochs):
        #extact the image and label from dataloader
        for images, labels in train_loader:
            optimizer.zero_grad() #clears previous gradient
            outputs = model(images)#gets model from batch
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# image genre prediction
def predict_image(model, img_path, transform, class_names):
    img = Image.open(img_path) #loads the image
    img = transform(img).unsqueeze(0)  # Add batch dimension
    model.eval()
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        return class_names[predicted.item()]

# Main function
def main():
    # Set data directory and parameters
    wikiart = "temp"
    data_dir = os.path.join(wikiart, 'dataset')
    test_image_path = 'adolf-fleischmann_hommage-delaunay-et-gleizes-1938.jpg'

    # Modify these numbers as needed
    batch_size = 128
    num_epochs = 10
    learning_rate = 0.001

    # Model file path
    model_path = "genre_classifier_cnn.pth"

    # Load data and initialize model
    train_loader, class_names = get_data_loader(data_dir, batch_size=batch_size)
    # set the number of the class name
    model = GenreClassifierCNN(num_classes=len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Check if a trained model exists
    if os.path.exists(model_path):
        # Load the model into the already savved model
        model.load_state_dict(torch.load(model_path))
        print("Model loaded from training file.")
    else:
        # Train and save the model if it doesn't exist
        print("Starting training...")
        train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)
        #save the model into it
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}.")

    # Define the transformation for prediction
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Test prediction on a single image
    predicted_genre = predict_image(model, test_image_path, transform, class_names)
    print(f"The genre is: {predicted_genre}")

# Run the main function
if __name__ == "__main__":
    main()
