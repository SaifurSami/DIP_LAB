# https://subscription.packtpub.com/book/data/9781784391454/1/ch01lvl1sec16/logarithmic-transformations
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

image_path = "cat.jpg"


def convert_to_gray(image):
    h, w, c = image.shape
    gray_image = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            b, g, r = image[i, j]
            gray_value = 0.3 * r + 0.59 * g + 0.11 * b
            gray_image[i, j] = int(gray_value)

    return gray_image


def main():
    image = cv2.imread(image_path)
    image = convert_to_gray(image)
    image = cv2.resize(image, (512, 512))
    normalized_image = np.copy(image).astype(np.uint8)
    normalized_image = normalized_image / 255.0

    gamma = 2
    power_image = np.uint8(np.power(normalized_image, gamma) * 255)
    scaling_factor = 255.0 / np.log(256)
    inverse_image = np.uint8((np.exp(image / scaling_factor) - 1))

    '''difference_image = power_image-inverse_image
    plt.imshow(difference_image,cmap='gray',vmin=0,vmax=255)
    plt.show()'''

    plt.figure(figsize=(10, 6))
    plt.subplot(3, 2, (1,2))
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.title("Original Image")

    plt.subplot(323)
    plt.imshow(power_image, cmap="gray", vmin=0, vmax=255)
    plt.title("Power Law")

    plt.subplot(324)
    plt.plot(image, power_image, "red")
    plt.title(f"Power Law Transformation(Gamma = {gamma})")

    plt.subplot(325)
    plt.imshow(inverse_image, cmap="gray", vmin=0, vmax=255)
    plt.title("Inverse Logarithm")

    plt.subplot(326)
    plt.plot(image, inverse_image, "blue")
    plt.title(f"Inverse Log Transformation(c = {scaling_factor})")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
