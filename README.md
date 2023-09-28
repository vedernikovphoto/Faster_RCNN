# Running Inference on a New Image with the Pre-trained Model

<div align="center">
    <img src="example.jpg" alt="Alternative Text">
</div>

<br><br>

Welcome to this repository! If you're eager to test the pre-trained model on your own image, follow the steps below:

## Prerequisites:

1. Ensure you have a Python environment with the necessary libraries installed. The required libraries and their respective versions are listed below:

    - `matplotlib==3.6.2`
    - `numpy==1.23.4`
    - `Pillow==10.0.0`
    - `scikit-learn==1.1.3`
    - `torch>=1.13.1`
    - `torchvision==0.14.0`
    - `tqdm==4.64.1`

   You can install them by using the following command:
    ```bash
    pip install -r requirements.txt
    ```

2. While not compulsory, a CUDA-capable GPU is recommended for faster inference. Without a GPU, the code will default to running on the CPU which will be slower.

3. Download the [`model_weights_final.pth`](https://shorturl.at/juJL5) for the fine-tuned model, which contains the model's trained weights. Then place downloaded file in the same folder with `inference.py` file.

5. Download the weights for [`fasterrcnn_resnet50_fpn`](https://shorturl.at/hvHPW) model. 

6. Rename downloaded weights file to `fasterrcnn_resnet50_fpn_coco.pth` and place the it in the same folder with `inference.py` file.


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
    The image will display detected humans with blue bounding boxes around them and red bounding boxes around their heads.

Thank you for using this model! Feel free to raise any issues or contribute.


