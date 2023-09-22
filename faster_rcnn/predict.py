import torch
from tqdm import tqdm


def predict(model, data_loader, score_threshold=0.8):
    """
    Uses the given model to make predictions on the provided data_loader
    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    
    predictions = []
    images = []
    ground_truths = []

    with torch.no_grad():  # Disable gradient calculation because we are in the testing mode
        
        for (image, targets) in tqdm(data_loader):
            
            image = list(img.to(device) for img in image)  # Move images to device (GPU or CPU) that the model is on
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
