import glob
import random
import os
import shutil

old_directory = os.path.join("D:","workspace","datasets","universal_image_embeddings")
trn_directory = os.path.join("datasets","universal_image_embeddings","train")
val_directory = os.path.join("datasets","universal_image_embeddings","eval")

trn_imgs_par_class = 512
val_imgs_par_class = 128

folders = os.listdir(old_directory)
for folder in folders:
    filepaths = glob.glob(os.path.join(old_directory, folder, "**"))
    fps = random.sample(filepaths, trn_imgs_par_class+val_imgs_par_class)
    trn_fps = fps[0:trn_imgs_par_class]
    val_fps = fps[trn_imgs_par_class:]

    if not os.path.isfile(os.path.join(trn_directory, folder)):
        os.makedirs(os.path.join(trn_directory, folder))
    for fp in trn_fps:
        shutil.copy(fp, os.path.join(trn_directory, folder, fp.split("\\")[-1]))

    if not os.path.isfile(os.path.join(val_directory, folder)):
        os.makedirs(os.path.join(val_directory, folder))
    for fp in val_fps:
        shutil.copy(fp, os.path.join(val_directory, folder, fp.split("\\")[-1]))

