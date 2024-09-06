import cv2
import numpy as np
from skimage.segmentation import clear_border
from matplotlib import pyplot as plt
import openslide 
import os
import pandas as pd
import warnings
import histolab
from skimage.color import rgb2gray
from histolab.slide import Slide
from histolab.tiler import GridTiler
from histolab.masks import TissueMask
from histolab.masks import BiggestTissueBoxMask
from skimage.measure import label, regionprops
from skimage.filters import threshold_local, threshold_otsu
import mahotas as mh
from skimage.color import rgb2lab, rgb2luv
from scipy import stats
import warnings
from PIL import Image
from histolab.tiler import ScoreTiler
from histolab.scorer import NucleiScorer
from scipy.stats import kurtosis, skew
import glob
import time

start_time = time.time()

BASE_PATH = '/home/vad3/projects/def-veman3/ul-val-prj-criucpq-poc/Oncotech-WSI/Patches'
##PROCESS_PATH_WSI = os.path.join(BASE_PATH, 'projects/def-veman3/ul-val-prj-criucpq-poc/Oncotech-WSI')
## On doit creer un dossier Patch dans Oncotech-WSI

## This fonction take as argument the path of the folder and extract the patch of all wsi in the folder. Also, this fonction delete the patch that have nuclei thresolding less than 15%.
def extract_patch (folder_path):
    wsi_files = [f for f in os.listdir(folder_path) if f.endswith('.ndpi')]
    warnings.filterwarnings("ignore")
    for file in wsi_files:
        wsi_path = os.path.join(folder_path, file)
        slide = Slide(wsi_path, processed_path = BASE_PATH)
        slide_name = os.path.splitext(os.path.basename(wsi_path))[0]
        scored_tiles_extractor = ScoreTiler(
        scorer = NucleiScorer(),
        tile_size=(512, 512),
        level=0,
        check_tissue=True,
        tissue_percent=80.0,
        pixel_overlap=0, # default
        prefix=f"Patch3/{slide_name}/", 
        suffix=".png" 
        )
        summary_filename = f"{slide_name}.csv"
        SUMMARY_PATH = os.path.join(slide.processed_path, summary_filename)
        scored_tiles_extractor.extract(slide, report_path=SUMMARY_PATH)
        pathcsv= os.path.join(BASE_PATH, f"{slide_name}.csv")
        df = pd.read_csv(pathcsv)
        filenames_to_delete = df[df['score'] < 0.15]['filename'].apply(lambda x: x.replace(f"Patch3/{slide_name}/", ''))
        # Parcourir les noms de fichiers et supprimer les fichiers correspondants
        images_folder = os.path.join('/home/vad3/projects/def-veman3/ul-val-prj-criucpq-poc/Oncotech-WSI/Patches/Patch3', f"{slide_name}")
        for filename in filenames_to_delete:
            image_path = os.path.join(images_folder, filename)
            if os.path.exists(image_path):
                os.remove(image_path)
                


extract_patch ('/home/vad3/projects/def-veman3/ul-val-prj-criucpq-poc/Oncotech-WSI/WSIs/WSIs3')


def compute_texture_feature (folder_path, tile_size = 32):
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    warnings.filterwarnings("ignore")
    haralick_CDM = pd.DataFrame()
    for file in png_files:
        image_path = os.path.join(folder_path, file)
        image_name = os.path.splitext(file)[0]
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
          ##  patch = img.convert('RGB')
            ##Blue Channel - Hematoxilin channel
        cells=img[:,:,0]  #Blue channel. Image equivalent to grey image.
            ## Otsu inverted binary thresholding to binarize the image
        ret, img_bin = cv2.threshold(cells,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            ## 3x3 kernel with all elements set to 1, which is used for morphology operations.
        kernel = np.ones((3,3),np.uint8)
            ## Opening morphology operation (erosion followed by dilation) to eliminate image noise.
        opening = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,kernel, iterations = 2)         
            ## dilates the image to obtain the sure background region.
        sure_bg = cv2.dilate(opening,kernel,iterations=10)
            ## performs a distance transformation to obtain a distance map.
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
            ## performs thresholding to obtain the sure foreground region
        ret2, sure_fg = cv2.threshold(dist_transform,0.25*dist_transform.max(),255,0)
        sure_fg = np.uint8(sure_fg)  #Convert to uint8 from float
        # Finding unknown region
        unknown = cv2.subtract(sure_bg,sure_fg)
            ## Find all connected components in the image
        ret3, markers = cv2.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0
        ## performs watershed segmentation on the image using the markers previously obtained
        ## The Watershed algorithm is a classic image segmentation algorithm used primarily for separating different 
        ## objects in an image the Watershed algorithm treats an image as a topographical landscape, floods it from 
        ## given seed points or markers, and creates boundaries where waters from different markers would meet, thereby 
        ## segmenting the image.
        markers = cv2.watershed(img,markers)
        tile_size = tile_size
        tiles = [markers[x:x+tile_size, y:y+tile_size] for x in range(0,markers.shape[0],tile_size) for y in range(0,markers.shape[1],tile_size)]
            ## The cellular density was estimated in each of these tiles by segmenting nuclei and assigning a gray level to each of 
            ## these tiles based on the number of nuclei estimated in each tile.
        density_map = np.zeros_like(gray) # Create a density map the same size as the original image
        for i, tile in enumerate(tiles):
            num_nuclei = np.count_nonzero(tile != -1)
            gray_level = int(num_nuclei / (tile_size * tile_size) * 255)  # Normalize to range 0-255
            x = (i // (img.shape[0]//tile_size)) * tile_size
            y = (i % (img.shape[1]//tile_size)) * tile_size
            density_map[x:x+tile_size, y:y+tile_size] = gray_level

        texture_feature = mh.features.haralick(density_map)
        texture_feature_array = np.array(texture_feature).flatten().reshape(1, -1)
        df_patch = pd.DataFrame(texture_feature_array, columns=["ASM_D1",	"contrast_D1",	"correlation_D1",	"sum_of_squares_D1",	"IDM_D1",	"sum_average_D1",	"sum_variance_D1",	"sum_entropy_D1",	"Entropy_D1",	"difference_variance_D1",	"difference_entropy_D1",	"IMC1_D1",	"IMC2_D1",
"ASM_D2",	"contrast_D2",	"correlation_D2",	"sum_of_squares_D2",	"IDM_D2",	"sum_average_D2",	"sum_variance_D2",	"sum_entropy_D2",	"Entropy_D2",	"difference_variance_D2",	"difference_entropy_D2",	"IMC1_D2",	"IMC2_D2",
"ASM_D3",	"contrast_D3",	"correlation_D3",	"sum_of_squares_D3",	"IDM_D3",	"sum_average_D3",	"sum_variance_D3",	"sum_entropy_D3",	"Entropy_D3",	"difference_variance_D3",	"difference_entropy_D3",	"IMC1_D3",	"IMC2_D3",
"ASM_D4",	"contrast_D4",	"correlation_D4",	"sum_of_squares_D4",	"IDM_D4",	"sum_average_D4",	"sum_variance_D4",	"sum_entropy_D4",	"Entropy_D4",	"difference_variance_D4",	"difference_entropy_D4",	"IMC1_D4",	"IMC2_D4"])
            ##df_patch.insert(0, 'image_name', image_name) 
        haralick_CDM = haralick_CDM.append(df_patch, ignore_index=True)
            # Calcul des moyennes et création du nouveau DataFrame
        texture_mean = haralick_CDM.mean().to_frame().T
        texture_mean.columns = ['mean_'+col for col in haralick_CDM.columns]
        texture_median = haralick_CDM.median().to_frame().T
        texture_median.columns = ['median_'+col for col in haralick_CDM.columns]
        texture_var = haralick_CDM.var().to_frame().T
        texture_var.columns = ['var_'+col for col in haralick_CDM.columns]
        texture_kurtosis = haralick_CDM.kurtosis().to_frame().T
        texture_kurtosis.columns = ['kurtosis_'+col for col in haralick_CDM.columns]
        texture_skewness = haralick_CDM.skew().to_frame().T
        texture_skewness.columns = ['skewness_'+col for col in haralick_CDM.columns]
        all_feature = pd.concat([texture_mean, texture_median, texture_var, texture_kurtosis, texture_skewness], axis=1)
                       
    return all_feature



## ## In this code we browse all subfolders containing patches to calculate haralick features and save them in a csv file.

patches_dir = '/home/vad3/projects/def-veman3/ul-val-prj-criucpq-poc/Oncotech-WSI/Patches/Patch3'

# Use glob to get a list of all subfolders in patches_dir
folder_list = glob.glob(os.path.join(patches_dir, "*"))
# Initialize a list to store dataframes
df_list1 = []
df_list2 = []
df_list3 = []
df_wsi_name=[]
# Browse the folder list and apply the compute_texture_feature function to each folder.
for folder_path in folder_list:
    wsi_name = os.path.basename(folder_path)
    df = compute_texture_feature (folder_path, tile_size = 16)
    df_list1.append(df)
    df_wsi_name.append(wsi_name)
# Concatenate all dataframes into a single dataframe
all_features_df_scale_16 = pd.concat(df_list1, ignore_index=True)
name_wsi = df_wsi_name
wsi_name_series = pd.Series(name_wsi, name='wsi_name')
all_features_df_scale_16.insert(0, 'wsi_name', wsi_name_series)
folder_path = "/home/vad3/projects/def-veman3/ul-val-prj-criucpq-poc/Oncotech-WSI"
file_name = "all_features_df_scale_16.csv"
full_path = os.path.join(folder_path, file_name)
# Save the DataFrame to a csv file
all_features_df_scale_16.to_csv(full_path, index=False)

for folder_path in folder_list:
    df = compute_texture_feature (folder_path, tile_size = 32)
    df_list2.append(df)
# Concatenate all dataframes into a single dataframe
all_features_df_scale_32 = pd.concat(df_list2, ignore_index=True)
all_features_df_scale_32.insert(0, 'wsi_name', wsi_name_series)
folder_path = "/home/vad3/projects/def-veman3/ul-val-prj-criucpq-poc/Oncotech-WSI"
file_name = "all_features_df_scale_32.csv"
full_path = os.path.join(folder_path, file_name)
# Save the DataFrame to a csv file
all_features_df_scale_32.to_csv(full_path, index=False)

for folder_path in folder_list:
    df = compute_texture_feature (folder_path, tile_size = 64)
    df_list3.append(df)
# Concatenate all dataframes into a single dataframe
all_features_df_scale_64 = pd.concat(df_list3, ignore_index=True)
all_features_df_scale_64.insert(0, 'wsi_name', wsi_name_series)
folder_path = "/home/vad3/projects/def-veman3/ul-val-prj-criucpq-poc/Oncotech-WSI"
file_name = "all_features_df_scale_64.csv"
full_path = os.path.join(folder_path, file_name)
# Save the DataFrame to a csv file
all_features_df_scale_64.to_csv(full_path, index=False)

end_time = time.time()
execution_time = end_time - start_time

print(f"Code execution time is : {execution_time} secondes.")