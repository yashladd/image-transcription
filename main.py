import os
from skimage import io
from skimage.filters import median, threshold_otsu
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.transform import resize_local_mean
import numpy as np
from predict import CharacterPredictor

PARAMS = {
    "msg_from_annie.png": {
        "square": 3,
        "area": 100,
    },
    "noisy_one_paragraph.jpg": {
        "square": 8,
        "area": 80,
    },
    "noisy_one_sentence.jpg": {
        "square": 3,
        "area": 100,
    },
    "noisy_three_sentences.jpg": {
        "square": 3,
        "area": 100,
    },
    "utes.png": {
        "square": 3,
        "area": 100,
    }
}

class Image(object):
    def __init__(self, file_name) -> None:
        self.file_name = file_name
        self.image = io.imread(os.path.join("test_images", self.file_name), as_gray=True)
        

def read_images():
    img_list = []
    cwd = os.getcwd()
    image_names = os.listdir(cwd + "/test_images")
    for f_name in image_names:
        img_list.append(Image(f_name))
    return img_list
    
def get_ordering(regions):
    ordered = []
    num_chars = len(regions)
    sorted_regions = sorted(regions, key=lambda x: x.bbox[0] + x.bbox[1])
    while len(ordered) < num_chars and len(sorted_regions):
        partial = []
        first_line_char = sorted_regions[0]
        centroid = first_line_char.centroid
        c_x, _ = centroid
        xmin, _, xmax, _ = first_line_char.bbox
        x_thr = c_x + abs(xmax - xmin)//2 + 10
        pos = []
        for i in range(len(sorted_regions)):
            region = sorted_regions[i]
            cntr_x, _ = region.centroid
            if cntr_x <= x_thr:
                partial.append(region)
                c_x, _ = region.centroid
                pos.append(i)
        for i in sorted(pos, reverse=True):
            del sorted_regions[i]  
        ordered.extend(sorted(partial, key=lambda x: x.centroid[1]))   
    return ordered 


def get_image_crops(regions, labelled_image):
    img_list = []
    for r in regions:
        bbox = r.bbox
        crop = labelled_image[bbox[0]-10: bbox[2]+10, bbox[1]-10:bbox[3]+10]
        img_list.append(crop)
    return img_list


def write_output(charachters, file_name):
    predictions = []
    predictor = CharacterPredictor(model_path="model.pth")
    for ch in charachters:
        predictions.append(predictor.predict(ch))
    text = "".join(predictions)
    name = file_name.split(".")[0]
    with open(f"./output/{name}.txt", "w+") as f:
        f.write(text)
    


if __name__ == "__main__":
    image_list = read_images()

    print("Images Loaded\n\n")
    current_working_directory = os.getcwd()
    # Make output directory for saving output images
    output_dir = current_working_directory + '/output'
    if not os.path.exists(output_dir):
        print("Creating output directory...\n\n")
        os.makedirs(output_dir)

    for img_obj in image_list:
        file_name = img_obj.file_name
        print(f"<------- Working on {file_name}, Please wait ------>")
        image = img_obj.image
        image = median(image)
        threshold = threshold_otsu(image)
        binary_image = np.where(image < threshold, 1, 0)


        closed = morphology.closing(binary_image, morphology.square(PARAMS[file_name]["square"]))
        labelled_image, count = measure.label(closed, return_num=True)

        region_properties = measure.regionprops(labelled_image)
        filtered_regions = []
        for r in region_properties:
            if r.area >= PARAMS[file_name]["area"]:
                filtered_regions.append(r)


        ordered_regions = get_ordering(filtered_regions)

        char_crops = get_image_crops(ordered_regions, labelled_image)
        resized_crops = [resize_local_mean(img, (28, 28)) for img in char_crops]
        charachters = [np.where(img > 0, 1, 0) for img in resized_crops]
        print(f"Char count in the image {file_name} is: {len(charachters)}")

        write_output(charachters, file_name)
        print(f"Text output written to output directory\n\n\n")