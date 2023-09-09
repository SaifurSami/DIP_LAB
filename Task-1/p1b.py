import cv2
import numpy as np
import matplotlib.pyplot as plt 
#import matplotlib.image as mpimg

image_path = 'dogs.jpg'

def convert_to_gray(image):
    h,w,c = image.shape
    gray_image = np.zeros((h,w),dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            b,g,r = image[i,j]
            gray_value = .3*r+.59*g+.11*b
            gray_image[i,j] = int(gray_value)
    gray_image
    return gray_image
def custom_intensity(image,bits):
    normalized_image = image / 255
    level = 2 ** bits
    normalized_image = np.uint8(np.floor(normalized_image*level))
    return normalized_image
def main():
    #img = cv2.imread(image_path)
    '''img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    plt.imshow(img)
    plt.axis('off')
    plt.show()'''
    image = cv2.imread(image_path)
    #image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = convert_to_gray(image) #convert the color image to gray image using luminosity method
    #print(image.shape)
    image = cv2.resize(image,(512,512))
    image_list = []
    image_list.append(image)
    for bits in range(7,0,-1):
        #print(bits)
        custom_image = custom_intensity(image,bits)
        image_list.append(custom_image)
    
    row,col = 2, 4
    fig,ax = plt.subplots(row,col,figsize=(10,6))
    indx = 0

    for i in range(row):
        for j in range(col):
            ax[i,j].imshow(image_list[indx],cmap='gray')
            ax[i,j].set_title(f'{8-indx} bit')
            indx += 1
    
    plt.tight_layout()
    plt.show()

    '''plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()'''
if __name__ == '__main__':
    main()




