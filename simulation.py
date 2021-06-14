from DNA import DNA
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

START_NUMBER = 1
GENERATED_PER_IMAGE = 5
IMG_SHAPE = (100, 100)
PREV_LOSS = 0
TRIANGLES = 1


def mutate_images(DNAs, mutation_per_image):
    """
    Generate new mutated images
    :param DNAs: List of images to make mutation from
    :param mutation_per_image: number of mutation to each image
    :return: list of images of length len(images)*(mutation_per_image+1)
    """
    mutated = []
    for dna in DNAs:
        mutated.append(dna)
        for _ in range(mutation_per_image):
            mutated.append(dna.mutate_n_triangles(TRIANGLES))
    return mutated


def get_n_best_images(DNAs, n, image):
    """
    Choose n most similar to the original photo DNAs
    :param DNAs:
    :param n: number of DNAs in return list
    :param image: original image
    :return: list of best DNAs
    """
    DNAs_loss = [(dna, dna.count_loss(image)) for dna in DNAs]
    sorted_DNAs_loss = sorted(DNAs_loss, key=lambda tup: tup[1])
    result = [DNAs_loss[0] for DNAs_loss in sorted_DNAs_loss[:n]]
    return result


def clear_saved_dir():
    files = os.listdir("savedDNA")
    for f in files:
        os.remove("savedDNA/" + f)


def load_dna(colors_path, coords_path):
    """
    Load a DNA from a saved npy array
    :param colors_path: location to load DNA.colors from
    :param coords_path: location to load DNA.coords from
    :return: DNA
    """
    colors = np.load(colors_path)
    coordinates = np.load(coords_path)
    dna = DNA(colors.shape[0])
    dna.colors = colors
    dna.coordinates = coordinates
    return dna


def run(iterations, pattern_path, name="Iter", step=100, initial_DNA=None):
    """
    Executes program to generate image, saves
    :param iterations: The number of iterations to be run
    :param pattern_path: the path to the pattern image
    :param name: The name of the output
    :param step: After each step the partial result is printed, and the current best DNA saved
    if <= 0, does not print partial results at all
    :param initial_DNA: The initial DNA, upon which the image would be generated (use "load_dna()")
    :return: The generated image after all iterations
    """
    #clear_saved_dir()

    im = Image.open(pattern_path)
    im = im.resize(IMG_SHAPE)
    source_image = np.array(im.convert('RGB'), dtype=np.uint8)

    if initial_DNA is not None:
        images_DNA = [initial_DNA for _ in range(START_NUMBER)]
        plt.imshow(images_DNA[0].generated_image(*IMG_SHAPE).astype(int))
        plt.show()
    else:
        images_DNA = [DNA(500) for _ in range(START_NUMBER)]
    for i in range(iterations):
        mutated_images = mutate_images(images_DNA, GENERATED_PER_IMAGE)
        images_DNA = get_n_best_images(mutated_images, START_NUMBER, source_image)

        # If step is positive, print and save results every step iterations
        if i % step == 0:
            print(i)
            print(images_DNA[0].count_loss(source_image))
            plt.imshow(images_DNA[0].generated_image(*IMG_SHAPE).astype(int))
            images_DNA[0].save(name + "_" + str(i))
            plt.show()


    cv2.imshow("Pattern Image", source_image)
    images_DNA[0].save(name + "_final")
    final_img = images_DNA[0].generated_image(*IMG_SHAPE).astype(np.uint8)
    plt.imshow(final_img)
    plt.show()
    cv2.imshow("Generated Image", final_img)
    #cv2.imwrite(name + ".png", final_img)

    while True:
        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            break
    return


if __name__ == '__main__':
    # load_dna("savedDNA/Marysia_final_colors.npy", "savedDNA/Marysia_final_coordinates.npy")
    run(
        40000,
        "Patterns/MonaLisa.jpg",
        "MonaLisaFromMarysia",
        500,
        DNA(1).load("MonaLisaFromMarysia_31500")
    )



