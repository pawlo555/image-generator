from DNA import DNA
from PIL import Image
import numpy as np


def visualize_image(location, height, width):
    """
    Loads DNA and generate image of set shape
    :param location: location of dna
    :param height: height of the generated image
    :param width: width of the generated image
    :returns: generated from loaded DNA PIL image
    """
    dna = DNA(0).load(location)
    image_array = dna.generated_image(height, width)
    return Image.fromarray(image_array.astype(np.uint8))


if __name__ == '__main__':
    image = visualize_image("monalisa_29900", 400, 200)
    image.save("result.png")
