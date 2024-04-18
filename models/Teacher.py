from typing import Union, List
from PIL import Image
import numpy as np
import torch
from transformers import ViTImageProcessor, AutoModelForImageClassification

class ImageConverter:
    @staticmethod
    def convert_to_pil(image, mode='RGB'):
        """
        Convert an image to a PIL Image object.

        Args:
            image (Union[str, np.ndarray, torch.Tensor]): The input image, can be a file path (str), a numpy array, or a PyTorch tensor.
            mode (str): The mode to convert the image to. Can be 'RGB' or 'RGBA'. Default is 'RGB'.

        Returns:
            PIL.Image.Image: The converted PIL Image object.
        """
        if isinstance(image, str):
            # If the image is a string (filepath), load the image
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            # If the image is a numpy array, convert it to a PIL image
            image = Image.fromarray(image)
        elif torch.is_tensor(image):
            # If the image is a PyTorch tensor, convert it to a numpy array first, then convert it to a PIL image
            image = Image.fromarray(image.cpu().permute(1, 2, 0).numpy())
        
        # Convert the image to the desired mode ('RGB' or 'RGBA')
        image = image.convert(mode)
        
        return image


class TeacherModel:
    """
    A class to represent the teacher model for image classification.

    Attributes:
        processor (transformers.AutoImageProcessor): The image processor.
        model (transformers.AutoModelForImageClassification): The image classification model.

    Methods:
        predict(image): Make a prediction on an image.
        batch_predict(images): Make predictions on a batch of images.
    """
    def __init__(self, model_name: str):
        """
        Initialize the TeacherModel with a specific model.

        Args:
            model_name (str): The name of the model to use.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        
        self.model = self.model.to(self.device)
        
        
    def one_hot_encode(self, output):
        """
        Convert the output to one-hot encoded format.

        Args:
            output (transformers.modeling_outputs.ImageClassifierOutput): The output from the model.

        Returns:
            list: The one-hot encoded output.
        """
        # Initialize the one-hot encoded output
        one_hot_output = [0, 0]

        # Apply softmax to the logits to get the probabilities
        probabilities = torch.nn.functional.softmax(output.logits, dim=1).detach().numpy()[0]

        # The index of the maximum probability is the predicted class
        predicted_class = np.argmax(probabilities)

        # Set the element at the index of the predicted class to 1
        one_hot_output[predicted_class] = 1

        return one_hot_output
    
    def batch_one_hot_encode(self, output):
        """
        Convert the output to one-hot encoded format.

        Args:
            output (torch.Tensor): The output logits from the model.

        Returns:
            torch.Tensor: The one-hot encoded output.
        """
        # Apply softmax to the logits to get the probabilities
        probabilities = torch.nn.functional.softmax(output, dim=0)

        # The index of the maximum probability is the predicted class
        _, predicted_class = torch.max(probabilities, dim=0)

        # Create a tensor for the one-hot encoded output
        one_hot_output = torch.zeros_like(probabilities)

        # Set the element at the index of the predicted class to 1
        one_hot_output.scatter_(0, predicted_class.unsqueeze(0), 1)

        return one_hot_output



    
    
    def predict(self, image: Union[str, Image.Image, np.ndarray, torch.Tensor]):
        """
        Make a prediction on an image.

        Args:
            image (Union[str, Image.Image, np.ndarray, torch.Tensor]): The input image, can be a file path (str), a PIL Image object, a numpy array, or a PyTorch tensor.

        Returns:
            dict: The prediction result from the model.
        """
        image = ImageConverter.convert_to_pil(image)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return self.one_hot_encode(outputs)



    def batch_predict(self, images: Union[List[str], List[Image.Image], List[np.ndarray], List[torch.Tensor]]):
        """
        Make predictions on a batch of images.

        Args:
            images (Union[List[str], List[Image.Image], List[np.ndarray], List[torch.Tensor]]): The input images, can be a list of file paths (str), PIL Image objects, numpy arrays, or PyTorch tensors.

        Returns:
            List[dict]: The prediction results from the model for each image.
        """
        # TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
        
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        return torch.stack([self.batch_one_hot_encode(logit) for logit in logits])

