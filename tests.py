from DNA import DNA
import cv2


dna = DNA(500)
image = dna.generated_image(400, 400)
mutated = dna.mutate()
mutated_image = mutated.generated_image(400, 400)
print("Image shape:")
print(image.shape)

cv2.imshow("image", image)
cv2.imshow("mutated", mutated_image)

while True:
    k = cv2.waitKey(1)
    if k == 27:
        cv2.destroyAllWindows()
        break
