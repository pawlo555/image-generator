from DNA import DNA
import cv2
IMG_SHAPE = (400, 400)

dna = DNA(50)
image = dna.generated_image(*IMG_SHAPE)
mutated = dna.mutate()
mutated_image = mutated.generated_image(*IMG_SHAPE)
print("Image shape:")
print(image.shape)

cv2.imshow("image", image)
cv2.imshow("mutated", mutated_image)

while True:
    k = cv2.waitKey(1)
    if k == 27:
        cv2.destroyAllWindows()
        break
