""" 
defines predictor class
"""

import string
from typing import Union, Optional
from numpy import ndarray
from torch import Tensor, load, from_numpy, unsqueeze, float32
from torch import device as _device
from torch.nn import Conv2d
from torchvision.models import resnet50


class CharacterPredictor:
    """Initialize a CharacterPredictor instance with a pretrained model

    Parameters
    ----------
    model_path : str
        Model file name, e.g. `model.pth`.
    device : str, torch.device, optional
        device where model will be ran on. default "cpu".

    """

    def __init__(self, model_path: str, device: Optional[Union[str, _device]] = "cpu"):
        self.device = device

        # initialize model
        self.model = resnet50()
        self.model.conv1 = Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.load_state_dict(load(model_path, map_location="cpu"))
        self.model.to(self.device)

        self.model.eval()

    def predict(self, char_image: Union[ndarray, Tensor]):
        """Predict a character from an image.

        Parameters
        ----------
        char_image : np.ndarray or torch.Tensor
            28x28 image of a character with 0 background (black)

        Returns
        -------
        predicted_char : str
            Predicted output of model as a capital character
            e.g. "A".

        """

        # check if char_image is of the right type and shape
        if isinstance(char_image, ndarray):
            char_image = from_numpy(char_image)
        elif not isinstance(char_image, Tensor):
            raise TypeError(
                f"expected image of type np.ndarray instead got {type(char_image)}"
            )
        if char_image.shape != (28, 28):
            raise ValueError(
                f"expected image of shape (28, 28), instead got {char_image.shape}"
            )

        # convert to 4D (batch, channels, x, y) and change type
        char_image = unsqueeze(unsqueeze(char_image, 0), 0)
        char_image = char_image.to(device=self.device, dtype=float32)

        # make prediction
        pred = self.model(char_image)
        pred = pred.argmax(1)
        predicted_char = list(string.ascii_uppercase)[pred]

        return predicted_char


if __name__ == "__main__":
    # This is a demo of the usage of CharacterPredictor.
    import os
    from skimage import io

    # create an instance and point to the pretrained model file
    predictor = CharacterPredictor(model_path="model.pth")

    # get example image filepaths
    example_images = os.listdir("example_images")
    example_images.sort()

    for example in example_images:
        # load the example image as grayscale
        image = io.imread(os.path.join("example_images", example), as_gray=True)
        # use the CharacterPredictor.predict() function to get the character written
        print(f"{example}: {predictor.predict(image)}")
