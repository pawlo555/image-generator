import cv2


def bgr2rgb(filename):
    """
    Converts an image from disc from brg format to rgb.
    :param filename: file location
    """
    loaded = cv2.imread(filename)
    changed = cv2.cvtColor(loaded, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, changed)


if __name__ == '__main__':
    bgr2rgb("kopernik10k.png")
