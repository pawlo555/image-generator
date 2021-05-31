from DNA import DNA
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

IMG_SHAPE = (100, 100)
START_NUMBER = 5
GENERATED_PER_IMAGE = 15
POPULATION_SIZE = 16
NO_TRIANGLES = 10



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

def select_roulette(DNAs, pattern_img, wheel=(4, 3, 2, 1)):
    """
    Selects new population based on the roulette wheel
    :param pattern_img: image to compare
    :param DNAs: DNA population of images
    :param wheel: The tuple with probability proportions
    :return: Dominant half of DNA, based on the roulette
    """
    return get_n_best_images(DNAs, len(DNAs) // 2, pattern_img)

def crossover(DNAs):
    random.shuffle(DNAs)
    new_population = DNAs
    for i in range(len(DNAs) // 2):
        new_population = np.concatenate([new_population, DNAs[2*i].mate(DNAs[2*i + 1])], axis=0)
    if len(DNAs) % 2 == 1:
        new_population = np.concatenate(
            [new_population, DNAs[-1].mate(DNAs[random.randrange(0, len(DNAs))])], axis=0
        )
    return new_population

def mutate(DNAs, args):
    pass



def run():
    im = Image.open('kopernik.jpg')
    im = im.resize(IMG_SHAPE)
    source_image = np.array(im.convert('RGB'), dtype=np.uint8)

    population = [DNA(NO_TRIANGLES) for _ in range(POPULATION_SIZE)]
    best = None

    for i in range(1000):
        print("Step {}".format(i))
        population = select_roulette(population, source_image)
        population = crossover(population)
        if i % 100 == 99:
            best = get_n_best_images(population, 1, source_image)
            plt.imshow(best[0].generated_image(*IMG_SHAPE).astype(int))
            plt.show()


    # for i in range(3):
    #     print(i)
    #     mutated_images = mutate_images(images_DNA, GENERATED_PER_IMAGE)
    #     images_DNA = get_n_best_images(mutated_images, START_NUMBER, source_image)
    #     plt.imshow(images_DNA[0].generated_image(*IMG_SHAPE).astype(int))
    #     plt.show()



    cv2.imshow("image", source_image)
    best = get_n_best_images(population, 1, source_image)
    final_img = best[0].generated_image(*IMG_SHAPE).astype(np.uint8)
    plt.imshow(final_img)
    plt.show()
    cv2.imshow("mutated", final_img)
    cv2.imwrite("result.png", final_img)

    while True:
        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            break
    return


if __name__ == '__main__':
    run()