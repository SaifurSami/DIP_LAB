import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'dogs.jpg'

def convert_to_gray(image):
    h,w,c = image.shape
    gray_image = np.zeros((h,w),dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            b,g,r = image[i,j]
            gray_value = 0.3*r + 0.59*g + 0.11*b
            gray_image[i,j] = int(gray_value)
    
    return gray_image
def histogram(image):
    fre = np.zeros(256,dtype=int)
    h,w = image.shape
    for i in range(h):
        for j in range(w):
            fre[image[i,j]] += 1
    return fre
    

def main():
    image = cv2.imread(image_path)
    image = convert_to_gray(image)
    image = cv2.resize(image,(512,512))
    '''plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.show()'''
    '''cnt = 0
    for i in range(256):
        print(frequency_list[i])
        cnt += frequency_list[i]
    print(cnt)'''
    frequency_list = histogram(image)
    

    threshold = input("Enter the Threshold = ")
    #print(threshold)
    colors = ['blue' if val != int(threshold) else 'red' for val in range(256)]
    segmented_image = (image > int(threshold)).astype(np.uint8)*255
    plt.figure(figsize=(10,6))
    plt.subplot(1,3,1)
    plt.imshow(image,cmap='gray',vmin=0,vmax=255)
    plt.title('Original Gray Image')
    plt.subplot(1,3,2)
    plt.bar(range(256), frequency_list, color=colors)
    plt.title('Histogram')
    plt.subplot(1,3,3)
    plt.imshow(segmented_image,cmap='gray',vmin=0,vmax=255)
    plt.title(f'Threshold : {threshold}')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()