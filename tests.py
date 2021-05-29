from DNA import DNA
import cv2


dna = DNA(50)
image = dna.generated_image(400, 400)
print("Image shape:")
print(image.shape)

cv2.imshow("image", image)

while True:
    k = cv2.waitKey(1)
    if k == 27:
        cv2.destroyAllWindows()
        break
