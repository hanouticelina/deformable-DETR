#===================================================
#                  COCO Extraction
#====================================================
import json
import numpy as np

def extract_coco(inFile, outFile,subset):
    """Function to extarct a subset from coco
    Parameters:
            infile: annotation file (json)
            outfile: output annotation file (json)
            subset: list of selected class ids
            percentage: percentage of total images to consider
    """
    with open(inFile) as f: #loading json annotation file
        data = json.load(f)
    """Nb_max = int(len(data["images"]) *percentage) # number of images to keep
    image_ids = []
    images = data["images"][: Nb_max] # list of kept images
    for i in images: # list of kept image ids
        image_ids.append(i["id"])"""
    images_ids = set()    
    annots = [] 
    for annot in data["annotations"]: # filter the annotations
        if annot["category_id"] in subset: # keep the ones about kept images
            annot["category_id"] = subset.index(annot["category_id"]) +1 
            annots.append(annot) # and of categorie in subset
            images_ids.add(annot["image_id"])
            
            
    images = [] # list of kept images
    for i in data["images"]: # list of kept image ids
        if i["id"] in images_ids:
            images.append(i)
            
    categories = [] # filter the categories
    for elem in data["categories"]:
        if elem["id"] in subset:
            elem["id"] = subset.index(elem["id"]) +1 
            categories.append(elem)
    
    
    data["annotations"] = annots # update the data
    data["images"] = images
    data["categories"] = categories
    
    with open(outFile, 'w') as data_file: # dump it in the outFile
        data = json.dump(data, data_file)
    return
    
def extract_coco_with_same_ids(inFile, outFile,subset):
    """Function to extarct a subset from coco
    Parameters:
            infile: annotation file (json)
            outfile: output annotation file (json)
            subset: list of selected class ids
            percentage: percentage of total images to consider
    """
    with open(inFile) as f: #loading json annotation file
        data = json.load(f)
    """Nb_max = int(len(data["images"]) *percentage) # number of images to keep
    image_ids = []
    images = data["images"][: Nb_max] # list of kept images
    for i in images: # list of kept image ids
        image_ids.append(i["id"])"""
    images_ids = set()    
    annots = [] 
    for annot in data["annotations"]: # filter the annotations
        if annot["category_id"] in subset: # keep the ones about kept images 
            annots.append(annot) # and of categorie in subset
            images_ids.add(annot["image_id"])
            
            
    images = [] # list of kept images
    for i in data["images"]: # list of kept image ids
        if i["id"] in images_ids:
            images.append(i)
            
    categories = [] # filter the categories
    for elem in data["categories"]:
        if elem["id"] in subset:
            categories.append(elem)
    
    
    data["annotations"] = annots # update the data
    data["images"] = images
    data["categories"] = categories
    
    with open(outFile, 'w') as data_file: # dump it in the outFile
        data = json.dump(data, data_file)
    return
    
# ===============================
# COCO light
# ===============================
# train
inFile = f"/tempory/coco/annotations/instances_train2017.json"
outFile = f"./coco_light/coco_light_train_ids.json"
subset = [23, 86, 6, 70, 32]
# test
extract_coco_with_same_ids(inFile, outFile, subset)

inFile = f"/tempory/coco/annotations/instances_val2017.json"
outFile = f"./coco_light/coco_light_validation_ids.json"
extract_coco_with_same_ids(inFile, outFile, subset)
