from DNA import DNA
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


source_path = "Patterns/WholeThing.jpg"
IMG_SHAPE = (100, 100)
POPULATION_SIZE = 60
NO_TRIANGLES = 20            # Should be even


def loadImage(path):
    im = Image.open(path).resize(IMG_SHAPE)
    return np.array(im.convert('RGB'), dtype=np.uint8)


def plotPopulation(population):
    for i in range(len(population)):
        plt.imshow(population[i].generated_image(*IMG_SHAPE).astype(int))
        plt.show()


def plotBestN(population, n, pattern_img):
    dominant = get_n_best_images(population, n, pattern_img)[0]
    plotPopulation(dominant)


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
    result = [DNAs_loss[0] for DNAs_loss in sorted_DNAs_loss[:n]]
    return result, sorted_DNAs_loss[0][1]


def select_roulette(DNAs, pattern_img, wheel=(4, 3, 2, 1)):
    """
    Selects new population based on the roulette wheel
    :param pattern_img: image to compare
    :param DNAs: DNA population of images
    :param wheel: The tuple with probability proportions
    :return: Dominant half of DNA, based on the roulette, and the best score
    """
    return get_n_best_images(DNAs, len(DNAs) // 2, pattern_img)


def crossover(DNAs):
    """
    Performs Crossover operator across population
    :param DNAs: populations' DNA
    :return: New population doubled in size
    """
    random.shuffle(DNAs)
    new_population = DNAs
    for i in range(len(DNAs) // 2):
        new_population = np.concatenate([new_population, DNAs[2*i].mate(DNAs[2*i + 1])], axis=0)
    if len(DNAs) % 2 == 1:
        new_population = np.concatenate(
            [new_population, DNAs[-1].mate(DNAs[random.randrange(0, len(DNAs))])], axis=0
        )
    return new_population


def mutateV2(DNAs, frequency, intensity):
    """
    Performs Mutation across the population
    :param DNAs: population genome
    :param frequency: the probability of mutation
    :param intensity: the proportion of genome to be modified
    """
    for dna in DNAs:
        if random.random() < frequency:
            dna.mutateV2(intensity)
    return


def mutateV1(DNAs, coords=0.03, colors=0.03):
    """
    Performs Mutation across the population
    :param colors: the difference in coords
    :param coords: the difference in colors
    :param DNAs: population genome
    """
    for dna in DNAs:
        dna.mutate(coords, colors)
    return


def run(maxIter):

    source_image = loadImage(source_path)
    population = [DNA(NO_TRIANGLES) for _ in range(POPULATION_SIZE)]
    score = 1000000000
    # plotPopulation(population)

    for i in range(maxIter):
        if i % 50 == 0:
            plotBestN(population, 1, source_image)
        population, currentScore = select_roulette(population, source_image)
        if currentScore < score:
            score = currentScore
            print("Step {}: loss={}".format(i, score))
        population = crossover(population)
        mutateV2(population, 0.2, 0.2)
        mutateV1(population, 0.1, 0.1)
        # plotPopulation(population)
    print("End of Loop")

    plotBestN(population, 5, source_image)
    cv2.imshow("image", source_image)
    best = get_n_best_images(population, 10, source_image)[0]
    final_img = best[0].generated_image(*IMG_SHAPE).astype(np.uint8)
    plt.imshow(final_img)
    plt.show()
    cv2.imshow("mutated", final_img)
    cv2.imwrite("kopernik10k.png", final_img)

    while True:
        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            break
    return


if __name__ == '__main__':
    run(4000)
