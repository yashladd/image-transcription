# Project 5 – Image transcription

get the pretrained model here: https://www.dropbox.com/s/gt4ow6huohcvted/model.pth?dl=0

In this project you will transcribe a given image into text, it is pretty much the same algorithm as midterm exam question 4. Given a handwritten/typed text image, the letters of which are written separated from each other in different font styles, your program should output all characters in the same order of image and write them in a text file. In this assignment you are using a pretrained model to classify the input characters; thus, training is not necessary and you do not need GPU access. To this end, you need to firstly extract the characters, then classify them, and finally write them in the order. The procedure can be regarded as follows.

1.	Extract individual characters
  - Preprocess by denoising the images. 
  -	Mask the image into foreground/background pixels using an authomatically defined threshold. 
  -	Extract connected components of the image. In this step you need to record the connected components which are the image characters. Also, since the order matters,  you need to compare the center of each connected region to track the relative locations of consecutive characters.
  
2.	Classifying characters 
  - Use the class given in `predict.py` to load a pretrained neural network and classify each character. 
    - Since the model input size is 28x28 and the connected components sizes are not necessarily the same, you need to pad the inputs accordingly before feeding them to the network. 
    - Remember to return the coordinates of each connected component in your dataloader so that you can preserve the order of characters.
    - `predict.py` contains some demo code showing you how to use the class. 

3.	Write the output
  - Your model outputs the recognized character, you still need the characters' order. 
  - Write the characters according to the order into a text file.


Download the starter code [here](.). The images you need to process are in `test_images/`. Files in `example_images/` are for the `predict.py` example code and do not need to be used in the assignment. 

## Code Submission Guidelines and Tips

Submit the code as a `.zip` file that contains
- A pdf report, including:
    - a description and discussion of your whole processing pipeline, including images where relevant
    - results on handwritten and typed images

- All the code, as required above, with comments.
- a `run.sh` script that runs your code, getting results for all text images. It will probably look something like this:
    ```bash
    #!/bin/bash

    python main.py
    ```

You're not being graded on code neatness, but that's not an excuse to not use good programming practices - writing reusable, readable code. 

