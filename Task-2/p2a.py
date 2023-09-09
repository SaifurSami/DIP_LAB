import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "dogs.jpg"


def convert_to_gray(image):
    h, w, c = image.shape
    gray_image = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            b, g, r = image[i, j]
            gray_value = 0.3 * r + 0.59 * g + 0.11 * b
            gray_image[i, j] = int(gray_value)

    return gray_image


def brightness_enhanchment(image, low, hi, factor):
    
    #modified_image = np.zeros((h,w),dtype=np.uint8)
    enhanched_image = np.copy(image)
    '''plt.imshow(enhanched_image)
    plt.show()'''
    h, w = enhanched_image.shape
    for i in range(enhanched_image.shape[0]):
        for j in range(enhanched_image.shape[1]):
            gray_value = enhanched_image[i,j]
            if gray_value >= int(low) and gray_value <= int(hi):
                gray_value = gray_value + int(factor)
                if gray_value > 255:
                    gray_value = 255
                elif gray_value < 0:
                    gray_value = 0
                enhanched_image[i,j] = gray_value
    
    return enhanched_image
                
    '''plt.imshow(enhanched_image)
    plt.show()
    return enhanched_image'''


def main():
    image = cv2.imread(image_path)
    image = convert_to_gray(image)
    image = cv2.resize(image, (512, 512))
    '''low= input("Enter the Range : ")
    high = input()
    enhanchment = input("Enter the Enhanchment = ")'''
    # print(low,high)
    low = input("Low = ")
    high = input("High = ")
    factor = input("Factor = ")
    #print(low,high,enhanchment)
    enhanched_image = brightness_enhanchment(image, low, high, factor)
    '''print(enhanched_image)
    plt.imshow(enhanched_image)
    plt.show()'''

    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.title("Original Image")
    plt.subplot(122)
    plt.imshow(enhanched_image, cmap="gray", vmin=0, vmax=255)
    plt.title(f"Enhanched between [{low}--{high}] by {factor}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
