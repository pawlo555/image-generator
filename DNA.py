import numpy as np
import cv2


"""
A DNA can be an array of floats with values from 0 to 1.
First 2 are the coordinates of first triangle points, second 2 for second point and third for third.
Then we will have a 3 values representing colors in triangle.
And for 100 triangles we have array containing 900 values.

To change DNA we can slightly change values of network.
Each value change by some randomly choice value for example in range(0.01, 0.1).
"""


class DNA:
    def __init__(self, triangles):
        """
        Creates a DNA which contain set parameters of triangles.
        :param triangles: number of triangles to generate
        """
        self.coordinates = np.random.random((triangles, 6))
        self.colors = np.random.random((triangles, 3))
        self.triangles = triangles

    def count_loss(self, image):
        """
        Counting a difference in input image and image generated from DNA
        :param image: input image - np.array of shape (height, width, 3)
        :return: loss of DNA
        """
        generated = self.generated_image(image.shape[0], image.shape[1])
        return np.sum(np.square(np.subtract(generated, image)))

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

    def get_triangles(self):
        """
        Get number of triangles coded in DNA
        :return: number of triangles in DNA
        """
        return self.triangles
