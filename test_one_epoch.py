from tqdm import tqdm
import torch


def test_one_epoch(model, data_loader, device):

    model.eval()

    ious = []  # IoUs of each prediction with the ground truth

    for images, targets in tqdm(data_loader, desc="Testing"):

        # Move the images and targets to the device (GPU or CPU)
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Disable gradient calculation since we run inference
        with torch.no_grad():
            outputs = model(images)

        for output, target in zip(outputs, targets):

            # Get the predicted boxes and ground truth boxes
            pred_boxes = output['boxes'].cpu().numpy()
            gt_boxes = target['boxes'].cpu().numpy()

            for pred_box in pred_boxes:
                iou_scores = []

                for gt_box in gt_boxes:

                    # Calculate the IoU of the predicted box and the ground truth box
                    iou = calculate_iou(pred_box, gt_box)
                    iou_scores.append(iou)

                max_iou_score = max(iou_scores)  # Get the maximum IoU score for the predicted box
                ious.append(max_iou_score)

    return sum(ious) / len(ious)


def calculate_iou(pred_box, gt_box):

    # Determine the coordinates of the intersection rectangle
    x1 = max(pred_box[0], gt_box[0])
    y1 = max(pred_box[1], gt_box[1])
    x2 = min(pred_box[2], gt_box[2])
    y2 = min(pred_box[3], gt_box[3])

    # Calculate the area of intersection rectangle
    width = (x2 - x1)
    height = (y2 - y1)

    # if there is no overlap, return 0
    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height

    # Calculate the combined area
    area_a = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    area_b = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    area_combined = area_a + area_b - area_overlap

    iou = area_overlap / (area_combined + 1e-6)  # Calculate the IoU

    return iou
