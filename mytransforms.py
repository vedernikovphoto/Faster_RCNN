from torchvision.transforms import functional as F


class MyTransforms:
    
    # train parameter indicates whether the transformations should be applied (applied if training)
    def __init__(self, train=True):
        self.train = train
    
    def __call__(self, image, target):
        image = F.to_tensor(image)  # Convert the image to a PyTorch tensor

        # Apply horizontal flip for training images
        if self.train:
            image = F.hflip(image)  # Apply a horizontal flip to the image
            target = self.flip_boxes(image, target)   # Adjust the bounding boxes of the target according to the flip

        return image, target


    @staticmethod
    def flip_boxes(image, target):
        """
        Flips bounding boxes when image is flipped
        """
        _, width = image.shape[-2:]
        boxes = target["boxes"]
        boxes[..., [0, 2]] = width - boxes[..., [2, 0]]  # Swap x_min and x_max
        target["boxes"] = boxes
        return target