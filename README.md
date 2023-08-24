# Running Inference on a New Image with the Pre-trained Model

Welcome to this repository! If you're eager to test the pre-trained model on your own image, follow the steps below:

## Prerequisites:

1. Ensure you have the necessary libraries. Install them with:
    ```bash
    pip install torch torchvision pillow matplotlib
    ```

2. CUDA-capable GPU is recommended for faster inference. Otherwise, the code will default to CPU.

3. Download the `model_weights_final.pth` from this repository, which contains the model's trained weights.

## Instructions:

1. **Clone the Repository**:
    ```bash
    git clone [YOUR_REPOSITORY_LINK]
    cd [YOUR_REPOSITORY_DIRECTORY]
    ```

2. **Prepare Your Image**:
    - If it doesn't exist, create an `Inference` folder in the root directory of the cloned repository.
    - Place your desired image inside this `Inference` folder. For this guide, we'll refer to this image as `test.jpg`.

3. **Modify Image Path**:
    Open the provided script in an editor and modify the following line:
    ```python
    image_path = '../Inference/7.jpg'
    ```
    Change to:
    ```python
    image_path = './Inference/test.jpg'
    ```

4. **Run the Script**:
    ```bash
    python [SCRIPT_NAME].py
    ```

5. **View Results**:
    Post execution, the image will be displayed with red outlined bounding boxes around detected objects.

Thank you for using this model! Feel free to raise any issues or contribute.

