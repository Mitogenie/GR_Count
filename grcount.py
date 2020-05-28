#GR count Core functions
#Last update 05/27/20
#Zong Lab @ University of Virginia

# Importing data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import re


if __name__ == '__main__':
    print('This is a helper file for GR count, and it is not supposed to run directly.')

# Set Address folder
def address():

    while True:
        address = input('Where are the files located?\n\n')

        try:
            file_list = os.listdir(address)
            return address, file_list
            break
        except:
            print('###########')
            print('Please make sure the address that you entered exists.')
            print('###########')

# Extract number of stack from the valid images
def extract_stacks(address,file_list):
    print('Please wait, while we are extrctaing stack info!')
    stacks = []
    file_names = []
    channels = []
    for filename in file_list:
        try:
            tmp = plt.imread(os.path.join(address,filename))
            tmp.shape
            file_names.append(filename)
            x = re.findall(r'(_z[0-9]+_c[0-9]+)', filename)[0]
            y = re.findall(r'(_c[0-9]+)', filename)[0]
            stack = x[2:x.find('c')-1]
            chl = y[2:]
            stacks.append(stack)
            channels.append(chl)

        except:
            pass
    file_names.sort()
    ch = int(max(channels))
    print('You are all set! You may now run the next cell.')
    return file_names, stacks,ch

def mask_it(address, test_images, Red_Th, Green_Th, RedBright, GreenBright, thresholds, image_shape):
    if 'c001' in test_images[0]:
        print('Currently looking at: ')
        print(test_images[0])
        print(test_images[1])
        img1 = cv2.imread(os.path.join(address,test_images[0]),cv2.IMREAD_GRAYSCALE)

        img11 = cv2.imread(os.path.join(address,test_images[0]))
        img2 = cv2.imread(os.path.join(address,test_images[1]),cv2.IMREAD_GRAYSCALE)
        img21 = cv2.imread(os.path.join(address,test_images[1]))
    else:
        print('Currently looking at: ')
        print(test_images[0])
        print(test_images[1])
        img1 = cv2.imread(os.path.join(address,test_images[1]),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img1)
        img11 = cv2.imread(os.path.join(address,test_images[1]))
        img2 = cv2.imread(os.path.join(address,test_images[0]),cv2.IMREAD_GRAYSCALE)
        img21 = cv2.imread(os.path.join(address,test_images[0]))

    th1, dst1 = cv2.threshold(img1, Red_Th, 255, cv2.THRESH_BINARY);
    th2, dst2 = cv2.threshold(img2, Green_Th, 255, cv2.THRESH_BINARY);

    f,ax = plt.subplots(2,2,figsize=(15,15))
    plt.subplot(221)
    plt.imshow(dst1,'gray')
    plt.title("Red Channel Mask")
    plt.subplot(222)
    plt.imshow(dst2,'gray')
    plt.title("Green Channel Mask")
    plt.subplot(223)
    plt.imshow(cv2.cvtColor(RedBright* img11, cv2.COLOR_BGR2RGB))
    plt.title("Red")
    plt.subplot(224)
    plt.imshow(cv2.cvtColor(GreenBright* img21, cv2.COLOR_BGR2RGB))
    plt.title("Green")
    thresholds[1] = Green_Th
    thresholds[0] = Red_Th
    image_shape = img21.shape
    return (thresholds[0], thresholds[1])

def remove_yellow(address, test_images,slide_of_iterest, Green_Red_ratio, thresholds,func_settings,non_coloc_mask_array):

    if 'c001' in test_images[0]:
        img1 = cv2.imread(os.path.join(address,test_images[0]),cv2.IMREAD_GRAYSCALE)
        img11 = cv2.imread(os.path.join(address,test_images[0]))
        img2 = cv2.imread(os.path.join(address,test_images[1]),cv2.IMREAD_GRAYSCALE)
        img21 = cv2.imread(os.path.join(address,test_images[1]))

    else:
        # print(test_images[0])
        img1 = cv2.imread(os.path.join(address,test_images[1]),cv2.IMREAD_GRAYSCALE)
        img11 = cv2.imread(os.path.join(address,test_images[1]))
        img2 = cv2.imread(os.path.join(address,test_images[0]),cv2.IMREAD_GRAYSCALE)
        img21 = cv2.imread(os.path.join(address,test_images[0]))

    img3 = img2/img1;
    where_are_NaNs = np.isnan(img3);
    where_are_infs = np.isinf(img3);
    img3[where_are_NaNs] = 0;
    img3[where_are_infs] = 0;
    img3 =((img3 < Green_Red_ratio) & (img3 >0));
    non_coloc_mask = 1 - img3;
    non_coloc_mask_transfered = non_coloc_mask;

    th1, dst1 = cv2.threshold(img1, thresholds[0], 255, cv2.THRESH_BINARY);
    th2, dst2 = cv2.threshold(img2, thresholds[1], 255, cv2.THRESH_BINARY);

    non_coloc_mask = np.bitwise_and(dst1, non_coloc_mask);
    non_coloc_mask = np.bitwise_and(dst2, non_coloc_mask);

    dst1_masked = dst1 - 255 * non_coloc_mask
    dst2_masked = dst2 - 255 * non_coloc_mask

    img_overlayed = np.zeros(img11.shape);
    img_overlayed[:,:,0] = np.multiply(dst1, img1);
    img_overlayed[:,:,1] = np.multiply(dst2, img2);

    img_overlayed_masked = np.zeros(img11.shape)
    img_overlayed_masked[:,:,0] = np.multiply(dst1_masked, img1);
    img_overlayed_masked[:,:,1] = np.multiply(dst2_masked, img2);

    f,ax = plt.subplots(2,2,figsize=(18,10))

    plt.subplot(231)
    plt.imshow(dst1, 'gray')
    plt.title("Red Channel Mask")

    plt.subplot(232)
    plt.imshow(5*img_overlayed)
    plt.title("Original Image With Red & Green Channels Overlaid")

    plt.subplot(233)
    plt.imshow(dst2, 'gray')
    plt.title("Green Channel Mask")

    plt.subplot(234)
    plt.imshow(dst1_masked, 'gray')
    plt.title("Red Channel Mask After Removing The Yellow Area")

    plt.subplot(235)
    plt.imshow(5*img_overlayed_masked)
    plt.title("Original Image With Overlaid Channels\nAfter Removing The Yellow Area")

    plt.subplot(236)
    plt.imshow(dst2_masked, 'gray')
    plt.title("Green Channel Mask After Removing The Yellow Area")

    func_settings[0] = slide_of_iterest;
    func_settings[1] = Green_Red_ratio;
    non_coloc_mask_array.append(non_coloc_mask_transfered);
    return non_coloc_mask_transfered

def measure_it(address,non_coloc_mask, file_names,ch,thresholds,measures=None):
    if measures is None:
        measures = []
    print('Please wait while we are measuring the ratios!')
    for i in range(0,len(file_names)-1,ch):

        try:
            img1 = cv2.imread(os.path.join(address,file_names[i]),cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(os.path.join(address,file_names[i+1]),cv2.IMREAD_GRAYSCALE)
            th1, dst1 = cv2.threshold(img1, thresholds[0], 255, cv2.THRESH_BINARY);
            th2, dst2 = cv2.threshold(img2, thresholds[1], 255, cv2.THRESH_BINARY);
            dst1 = np.bitwise_and(dst1, non_coloc_mask)
            dst2 = np.bitwise_and(dst2, non_coloc_mask)
            x = re.findall(r'(_z[0-9]+_c[0-9]+)', file_names[i])[0]
            stack = x[2:x.find('c')-1]
            measures.append((stack,dst2.sum(),dst1.sum(), (dst2.sum()/dst1.sum()),((dst1.sum()/dst2.sum()))))
        except:
            pass

    measures = pd.DataFrame(measures)
    measures.columns = [['Stack','Green Area', 'Red Area', 'Green/Red Ratio','Red/Green Ratio']]
    measures.replace(np.inf, np.nan, inplace=True)

    print('\n You are all set! You may now proceed to the next cell.')

    return measures

def plot_results(measurements_cleaned):
    plt.subplots(3,1,figsize=(16,16))

    plt.subplot(2,1,1)
    plt.plot(measurements_cleaned.iloc[:,1], label ='Sum Green', marker='.', color='green')
    plt.plot(measurements_cleaned.iloc[:,2], label ='Sum Red', marker='.', color='red')

    plt.legend(fontsize=14)
    plt.xlabel('Stack Number', fontsize=14)
    plt.ylabel('Signal sum for each channel',fontsize=14)
    plt.title('Signal Sum in Different Channels',fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)


    plt.subplot(2,1,2)
    plt.plot(measurements_cleaned.iloc[:,3:4], label ='Green/Red Ratio', marker='.')
    plt.legend(fontsize=14)
    plt.xlabel('Stack Number', fontsize=14)
    plt.ylabel('GR Ratio',fontsize=14)
    plt.title('GR Ratio',fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # plt.subplot(3,1,3)
    # plt.plot(measurements_cleaned.iloc[:,3:4], label ='Red/Green Ratio', marker='.')
    # plt.legend(fontsize=14)
    # plt.xlabel('Stack Number', fontsize=14)
    # plt.ylabel('RG Ratio',fontsize=14)
    # plt.title('RG Ratio',fontsize=18)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)

    plt.show()

def stack_sum(measurements):
    sum_data = measurements.sum(axis=0)
    sum_data = pd.DataFrame(sum_data).transpose()
    sum_data = sum_data.iloc[:,1:3]
    sum_data.columns = ['Total Green Area along the Stack','Total Red Area along the Stack']
    sum_data['Green/Red Ratio of the stack'] = sum_data['Total Green Area along the Stack'] / sum_data['Total Red Area along the Stack']
    sum_data['Red/Green Ratio of the stack'] = sum_data['Total Red Area along the Stack'] / sum_data['Total Green Area along the Stack']
    sum_data['Stack Results'] = '>>>'
    sum_data.set_index('Stack Results', inplace=True);
    return sum_data
