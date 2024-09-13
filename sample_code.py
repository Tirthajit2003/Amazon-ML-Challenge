import os
import random
import pandas as pd
import torch
from torchvision import models, transforms
from PIL import Image
from src.utils import download_image


def load_model():
    # Load a pre-trained ResNet model
    model = models.resnet18(pretrained=True)
    model.eval()  # Set the model to evaluation mode
    return model


def preprocess_image(img_path):
    # Define the transformations: resize, center crop, convert to tensor, normalize
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Open the image and apply transformations
    img = Image.open(img_path)
    img_t = transform(img)

    # Add a batch dimension for the model
    batch_t = torch.unsqueeze(img_t, 0)
    return batch_t


def predictor(image_link, category_id, entity_name, model):
    '''
    Predict using a pre-trained model
    '''
    # Folder to save the image
    save_folder = 'dataset/images/'

    # Check if the folder exists, if not, create it
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)  # Create the folder (and any intermediate folders)

    print("Entered")
    img_path = download_image(image_link, save_folder)
    print(img_path)

    if img_path:  # Check if the image is downloaded successfully
        try:
            # Preprocess the image
            img_tensor = preprocess_image(img_path)

            # Make a prediction using the pre-trained model
            with torch.no_grad():
                output = model(img_tensor)


        except Exception as e:
            print(f"Error processing image: {e}")
            return "Error processing image"

    return "" if random.random() > 0.5 else "10 inch"


if __name__ == "__main__":
    # Load the model once before making predictions
    model = load_model()

    DATASET_FOLDER = 'dataset/'

    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))

    # Pass the model to the predictor
    test['prediction'] = test.apply(
        lambda row: predictor(row['image_link'], row['group_id'], row['entity_name'], model), axis=1)

    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)
