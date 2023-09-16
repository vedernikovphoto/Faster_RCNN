# Running Inference on a New Image with the Pre-trained Model

<div align="center">
    <img src="example.jpg" alt="Alternative Text">
</div>

Welcome to this repository! If you're eager to test the pre-trained model on your own image, follow the steps below:

## Prerequisites:

1. Ensure you have a Python environment with the necessary libraries installed. You can install them by using the following command:

    - `matplotlib==3.6.2`
    - `numpy==1.23.4`
    - `Pillow==10.0.0`
    - `scikit-learn==1.1.3`
    - `torch==1.13.0`
    - `torchvision==0.14.0`
    - `tqdm==4.64.1`
   
    ```bash
    pip install -r requirements.txt
    ```

2. While not compulsory, a CUDA-capable GPU is recommended for faster inference. Without a GPU, the code will default to running on the CPU which will be slower.

<!--3. Download the weights for `fasterrcnn_resnet50_fpn` model from https://shorturl.at/hvHPW 

4. Rename downloaded weights file to `fasterrcnn_resnet50_fpn_coco.pth` and place the it in the same folder with `inference.py` -->

3. Download the `model_weights_final.pth` for fine-tuned model from https://shorturl.at/eJTU6, which contains the model's trained weights.

4. Place downloaded `model_weights_final.pth` in the same folder with `inference.py` 

## Instructions:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/vedernikovphoto/Faster_RCNN
    cd [YOUR_REPOSITORY_DIRECTORY]
    ```

2. **Prepare Your Image**:
    - If it doesn't exist, create an `Inference` folder in the root directory of the cloned repository.
    - Place your desired image inside this `Inference` folder. For this guide, we'll refer to this image as `test.jpg`.

3. **Run the Script**:
    ```bash
    python inference.py
    ```

4. **View Results**:
    Post execution, the image will be displayed with red outlined bounding boxes around detected objects.

Thank you for using this model! Feel free to raise any issues or contribute.


