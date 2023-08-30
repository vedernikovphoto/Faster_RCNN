import torch
from tqdm import tqdm


def predict(model, data_loader, score_threshold=0.8):
    """
    Uses the given model to make predictions on the provided data_loader
    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Set the model to evaluation mode
    model.eval()
    
    # Initialize lists to store the predictions, the images, the ground truth data and the indices.
    predictions = []
    images = []
    ground_truths = []

    # Disable gradient calculation because we are in the testing mode
    with torch.no_grad():
        
        # Iterate over each batch of data in the data loader
        for (image, targets) in tqdm(data_loader):
            
            # Move the images to the device (GPU or CPU) that the model is on
            image = list(img.to(device) for img in image)
            
            # Use the model to predict the targets from the images
            prediction = model(image)
            
            # Apply a threshold to the scores
            for pred in prediction:
                confident_indices = pred['scores'] > score_threshold
                pred['boxes'] = pred['boxes'][confident_indices]
                pred['labels'] = pred['labels'][confident_indices]
                pred['scores'] = pred['scores'][confident_indices]
            
            # Move the predictions and images back to CPU memory
            predictions.extend([{k: v.to('cpu') for k, v in pred.items()} for pred in prediction])
            images.extend([img.to('cpu') for img in image])
            
            # Store the ground truth data
            ground_truths.extend(targets)
            
    return predictions, images, ground_truths