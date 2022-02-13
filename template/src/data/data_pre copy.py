# -*- coding: utf-8 -*-

import tensorflow as tf
from loguru import logger
import h5py
import os
from tqdm import tqdm
import cv2

logger.add("../reports/debug.log")

def get_raw_data(
    
    datadir_raw,
    url: str = "https://astro.utoronto.ca/~hleung/shared/Galaxy10/Galaxy10_DECals.h5",  
) -> None:
    """ toelichting """

    if not datadir_raw.exists(): # check of directory al bestaat, zo niet maak
        datadir_raw.mkdir(parents=True)
        logger.info(f"Creating folder {datadir_raw}")
    if (datadir_raw / "Galaxy10_decals.h5").exists(): # check of dataset al aanwezig is, zo niet download
        logger.info(f"found Galaxy10_decals.h5 in {datadir_raw}, not downloading again")
    else:
        logger.info(f"Data not present in {datadir_raw}, downloading from url")
        datadir_raw.mkdir(parents=True, exist_ok=True)
        tf.keras.utils.get_file(
            origin=url,
            fname="Galaxy10_decals.h5",
            cache_dir=datadir_raw,
            cache_subdir="",
            extract=True,
        )

def sub_folders(
    datadir_processed,
    datadir_raw,
    galaxy_types

) -> None:
    
    hf = h5py.File(datadir_raw /"Galaxy10_decals.h5", 'r')
    
    if not datadir_processed.exists(): # check of directory al bestaat, zo niet maak
        datadir_processed.mkdir(parents=True)
        logger.info(f"Creating folder {datadir_processed}")
    else:
        logger.info(f"Main folder {datadir_processed} allready exists")
    for x in galaxy_types:
        subfolder = os.path.join(datadir_processed, x)

        if not (datadir_processed / x).exists():
            os.mkdir(subfolder)
            logger.info(f"Creating subfolder {subfolder}")
        else:
            logger.info(f"Subfolder {subfolder} allready exists")

def count_images(
         datadir_processed,
         printcount,
         galaxy_types,
) -> None:

    Total_images = 0
    
    for i in range(len(galaxy_types)):
        k = galaxy_types[i]
        save_folder = os.path.join(datadir_processed, k)
    
        for (root,dirs,files) in os.walk(save_folder, topdown=False):
        
            if len(files)>0:
                Nfiles=len(files)
                Total_images = Total_images + Nfiles
                if printcount == 'y':
                    print(k + ' ' + str(Nfiles))
    if printcount == 'y':
        print('Total number of images: ' + str(Total_images))
    count_images.Total = Total_images

def process_data(
    datadir_processed,
    datadir_raw,
    galaxy_types,
    images,
    labels,

) -> None:

    printcount = 'n'
    count_images(datadir_processed, printcount, galaxy_types)
     # to check if files allready exist
    rows = images.shape[0] # number of images in hf file

    hf = h5py.File(datadir_raw /"Galaxy10_decals.h5", 'r')
    
    with hf:

        if not count_images.Total==17736:
            logger.info(f"Unpacking images")
    #print(rows)
            for i in tqdm(range(hf['images'].shape[0])):
                img_arr = hf['images'][i,:]   
                img_arr_rs = cv2.resize(img_arr, (64, 64) ) # resize image to save space/memory

    
                k = galaxy_types[labels[i]] # get type of galaxy
        
        
                save_folder = os.path.join(datadir_processed, k) # determine correct subfolder
        
                cv2.imwrite(save_folder + '/' + f'img_{i:03}.jpg',img_arr_rs) # save image to correct subfolder

        else:
            logger.info(f"Data allready exists")

def get_size(datadir_processed
) -> None:
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(datadir_processed):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size/1000000


