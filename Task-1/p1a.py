import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = 'sample.jpg'

def downsample(img,scale):
    h,w = img.shape
    r_h,r_w = h // scale, w // scale
    new_img = np.zeros((r_h,r_w))

    for i in range(r_h):
        for j in range(r_w):
            new_img[i,j] = img[i*scale,j*scale]
    
    return new_img
def main():
    img_list = []
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #resizing the image to 512x512
    re_img = cv2.resize(img,(512,512))
    img_list.append(re_img)
    #print(re_img.shape)

    for _ in range(8):
        re_img = downsample(re_img,2)
        img_list.append(re_img)
    
    row, col = 2,4
    indx = 0
    fig,axes = plt.subplots(row,col,figsize=(9,7)) #2 rows & 4 columns for 8 subplots or 8 grids
    for i in range(row):
        for j in range(col):
            axes[i,j].imshow(img_list[indx],cmap='gray')
            h = img_list[indx].shape[0]
            w = h = img_list[indx].shape[1]
            axes[i,j].set_title(f'{h}x{w}')
            indx += 1
    plt.tight_layout() #for proper arrangement of the subplots
    plt.show()

if __name__ == '__main__':
    main()
