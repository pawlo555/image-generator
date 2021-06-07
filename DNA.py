import numpy as np
import cv2
from math import ceil, sqrt


"""
A DNA can be an array of floats with values from 0 to 1.
First 2 are the coordinates of first triangle points, second 2 for second point and third for third.
Then we will have a 3 values representing colors in triangle.
And for 100 triangles we have array containing 900 values.

To change DNA we can slightly change values of network.
Each value change by some randomly choice value for example in range(0.01, 0.1).
"""

def RGB_Distance(colors):
    """
    Metric in RGB color space
    :param pxl_1: (R, G, B)
    :param pxl_2: (R, G, B)
    :return: the distance between pixels
    """
    r1, g1, b1, r2, g2, b2 = colors[0], colors[1], colors[2], colors[3], colors[4], colors[5],
    r_mean = (r1 + r2) / 2
    d_r, d_g, d_b = r1 - r2, g1 - g2, b1 - b2
    return sqrt(((2 + (r_mean / 256))*d_r*d_r) + (4*d_g*d_g) + ((1/256)*(767 - r_mean)*d_b*d_b))

def count_loss(img1, img2):
    return np.sum(np.square(np.apply_along_axis(
        RGB_Distance, 2,
        np.concatenate([img1, img2], axis=2)
    )))

class DNA:
    def __init__(self, triangles):
        """
        Creates a DNA which contain set parameters of triangles.
        :param triangles: number of triangles to generate
        """
        self.coordinates = np.random.random((triangles, 6))
        self.colors = np.random.random((triangles, 3))
        self.triangles = triangles

    def create(self, coords, colors, triangles):
        result = DNA(0)
        result.coordinates = coords
        result.colors = colors
        result.triangles = triangles
        return result

    def count_loss(self, image):
        """
        Counting a difference in input image and image generated from DNA
        :param image: input image - np.array of shape (height, width, 3)
        :return: loss of DNA
        """
        generated = self.generated_image(image.shape[0], image.shape[1])
        return np.sum(np.square(np.subtract(generated, image)))

    def count_loss_V2(self, other):
        return np.sum(np.square(np.apply_along_axis(
            RGB_Distance, 2,
            np.concatenate([self.generated_image(other.shape[0], other.shape[1]), other], axis=2)
        ))) / 1000

    def generated_image(self, height, width):
        """
        Generate an image using actual DNA
        :param height: height of generated image
        :param width: width of generated image
        :return: image of shape (height, width, 3)
        """
        n = self.get_triangles()
        images = np.zeros(shape=(n, height, width, 3))
        mask = np.zeros(shape=(height, width, 3))

        for i in range(n):
            point1 = (self.coordinates[i, 0]*height, self.coordinates[i, 1]*width)
            point2 = (self.coordinates[i, 2]*height, self.coordinates[i, 3]*width)
            point3 = (self.coordinates[i, 4]*height, self.coordinates[i, 5]*width)
            contour = np.array([point1, point2, point3])

            cv2.drawContours(images[i], [contour.astype(int)], 0, self.colors[i]*255, -1)
            mask[images[i] != 0] = mask[images[i] != 0] + 1
        mask[mask == 0] = 1
        result = np.sum(images, axis=0)/mask
        return result

    def mutate(self, coordinates_diff=0.03, color_diff=0.03):
        """
        Generate new DNA based on self. Similarity to based image depends of coordinates_diff and color_diff parameters.
        :param coordinates_diff: Determine how much coordinates change.
        :param color_diff: Determine how much color change.
        :return: new DNA
        """
        new_DNA = DNA(self.get_triangles())
        for i in range(self.get_triangles()):
            new_DNA.coordinates[i] = self.coordinates[i] + (new_DNA.coordinates[i] - 0.5) * coordinates_diff
            new_DNA.colors[i] = self.colors[i] + (new_DNA.colors[i] - 0.5) * color_diff

        new_DNA.coordinates[new_DNA.coordinates < 0] = 0
        new_DNA.coordinates[new_DNA.coordinates > 1] = 1
        new_DNA.colors[new_DNA.colors < 0] = 0
        new_DNA.colors[new_DNA.colors > 1] = 1

        return new_DNA

    def mutateV2(self, intensity):
        """
        Mutates genes with given intensity
        :param intensity: the proportion of genes to be modified
        """
        indices = np.random.choice(
            range(self.triangles),
            ceil(intensity * self.triangles),
            replace=False)
        for idx in indices:
            self.coordinates[idx] = np.random.random(6)
            self.colors[idx] = np.random.random(3)

    def mate(self, other):
        """
        Two DNAs creates two children
        :param other: partner to mate with
        :return: list of two children
        """
        children = [DNA(self.triangles) for _ in range(2)]

        for id in range(2):
            indices = np.sort(np.random.choice(
                range(self.triangles),
                self.triangles // 2,
                replace=False))
            idx = 0
            for j in range(self.triangles):
                if idx < len(indices):
                    if indices[idx] == j:
                        children[id].coordinates[j] = self.coordinates[j]
                        children[id].colors[j] = self.colors[j]
                        idx += 1
                    else:
                        children[id].coordinates[j] = other.coordinates[j]
                        children[id].colors[j] = other.colors[j]
                    children[id].coordinates[j] = other.coordinates[j]
                    children[id].colors[j] = other.colors[j]
        return children


    def get_triangles(self):
        """
        Get number of triangles coded in DNA
        :return: number of triangles in DNA
        """
        return self.triangles

def myMax(t1, t2):
    return max(t1[0], t2[0]), max(t1[1], t2[1]), max(t1[2], t2[2])

if __name__ == '__main__':
    # c1 = (25, 0, 255)
    # c2 = (17, 0, 173)
    # print(RGB_Distance(c1, c2))

    a = np.ones((5, 5, 3))
    print(count_loss(a, np.zeros((5, 5, 3))))
