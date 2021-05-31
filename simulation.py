from DNA import DNA
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

START_NUMBER = 5
GENERATED_PER_IMAGE = 15
IMG_SHAPE = (100, 100)


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
            mutated.append(dna.mutate())
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
    print(sorted_DNAs_loss[0])
    result = [DNAs_loss[0] for DNAs_loss in sorted_DNAs_loss[:n]]
    return result


im = Image.open('kopernik.jpg')
im = im.resize(IMG_SHAPE)
source_image = np.array(im.convert('RGB'), dtype=np.uint8)

images_DNA = [DNA(50) for _ in range(START_NUMBER)]
for i in range(10):
    print(i)
    mutated_images = mutate_images(images_DNA, GENERATED_PER_IMAGE)
    images_DNA = get_n_best_images(mutated_images, START_NUMBER, source_image)
    plt.imshow(images_DNA[0].generated_image(*IMG_SHAPE).astype(int))
    plt.show()


cv2.imshow("image", source_image)
plt.imshow(images_DNA[0].generated_image(*IMG_SHAPE))
plt.show()
plt.savefig("result.png")
cv2.imshow("mutated", images_DNA[0].generated_image(*IMG_SHAPE))

while True:
    k = cv2.waitKey(1)
    if k == 27:
        cv2.destroyAllWindows()
        break
