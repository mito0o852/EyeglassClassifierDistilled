from models.Student import GlassesClassifier, BottleneckBlock
from scripts.data_preprocessing import ImageDatasetLoader, UnlabeledImageDataset
from models.Teacher import TeacherModel
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


# Dataset loading
dataset = UnlabeledImageDataset('dataset/train/train')
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


validation = ImageDatasetLoader('dataset/val')
val_loader = validation.load_images()


# Load the teacher model
teacher = TeacherModel("youngp5/eyeglasses_detection")

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Models
model = GlassesClassifier(BottleneckBlock, [3, 4, 6, 3], num_classes=2)
model = model.to(device)


# Set the optimizer and the learning rate scheduler
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

# Set the number of epochs
num_epochs = 10

# Initialize the best accuracy
best_accuracy = 0.0

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data) in enumerate(dataloader):
        
        data = data.to(device)
        data = data.float()
        if len(data.shape) == 3:
            data = data.unsqueeze(0)  # Adds a batch dimension at the beginning

        optimizer.zero_grad()
        
        # Forward pass of the student
        student_output = model(data)

        # Forward pass of the teacher
        teacher_output = teacher.batch_predict(data)

        # Calculate the loss
        loss = F.kl_div(F.log_softmax(student_output, dim=1), F.softmax(teacher_output, dim=1))

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

    print(f"Epoch: {epoch}, Loss: {loss.item()}")
    
    # Evaluation
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / len(val_loader.dataset)

    # Save the model if it's the best so far
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), "best_model.pth")

    print(f"Epoch: {epoch}, Loss: {loss.item()}, Accuracy: {accuracy}")



print(f"Best Accuracy: {best_accuracy}")
