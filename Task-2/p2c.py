import cv2 
import numpy as np
import matplotlib.pyplot as plt

image_path = 'cat.jpg'

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
    image = cv2.resize(image,(512,512))

    custom_image = np.copy(image).astype(np.uint8)
    custom_image = custom_image & 224

    difference_image = image - custom_image

    plt.figure(figsize=(10,6))
    plt.subplot(221)
    plt.imshow(image,cmap='gray',vmin=0,vmax=255)
    plt.title('Original Image')

    plt.subplot(222)
    plt.imshow(custom_image,cmap='gray',vmin=0,vmax=255)
    plt.title('MSB-3bits')

    plt.subplot(2,2,(3,4))
    plt.imshow(difference_image,cmap='gray',vmin=0,vmax=255)
    plt.title('Difference Image')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()