import os
import pandas as pd
import easyocr
from PIL import Image
import torch
from torchvision import models, transforms
from src.utils import download_image

# Initialize the OCR reader
reader = easyocr.Reader(['en'])

# Entity-unit mapping from the provided appendix
entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon',
                    'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
}


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


def extract_text_from_image(img_path):
    # Use OCR to extract text
    results = reader.readtext(img_path)
    extracted_text = " ".join([res[1] for res in results])
    return extracted_text


def match_entity_value(extracted_text, entity_name):
    # Extract number and unit based on the entity type
    allowed_units = entity_unit_map.get(entity_name, [])

    for unit in allowed_units:
        if unit in extracted_text:
            # Extract the number before the unit
            try:
                # Find the number that comes before the unit
                number = float(extracted_text.split(unit)[0].strip().split()[-1])
                return f"{number} {unit}"
            except:
                continue
    return ""


def predictor(image_link, category_id, entity_name, model):
    '''
    Predict using the model and OCR
    '''
    save_folder = 'dataset/images/'

    # Check if the folder exists, if not, create it
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)  # Create the folder (and any intermediate folders)

    print("Entered")
    img_path = download_image(image_link, save_folder)
    print(img_path)

    if img_path:  # Check if the image is downloaded successfully
        try:
            # Extract text from the image using OCR
            extracted_text = extract_text_from_image(img_path)
            print(f"Extracted Text: {extracted_text}")

            # Match the extracted text with the expected entity name and value
            predicted_value = match_entity_value(extracted_text, entity_name)
            print(f"Predicted Value: {predicted_value}")
            return predicted_value

        except Exception as e:
            print(f"Error processing image: {e}")
            return "Error processing image"

    return ""


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
