#===================================================
#                  COCO Extraction
#====================================================
import json
import numpy as np

def extract_coco(inFile, outFile,subset, percentage):
    """Function to extract a subset from coco
    Parameters:
            infile: annotation file (json)
            outfile: output annotation file (json)
            subset: list of selected class ids
            percentage: percentage of total images to consider
    """
    with open(inFile) as f: #loading json annotation file
        data = json.load(f)
    Nb_max = int(len(data["images"]) *percentage) # number of images to keep
    image_ids = []
    images = data["images"][: Nb_max] # list of kept images
    for i in images: # list of kept image ids
        image_ids.append(i["id"])
        
    annots = [] 
    for annot in data["annotations"]: # filter the annotations
        if annot["category_id"] in subset and annot["image_id"] in image_ids: # keep the ones about kept images
            annots.append(annot) # and of categorie in subset
            
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
outFile = f"./coco_light/coco_light_train.json"
percentage = 0.4
subset = np.random.choice(90, 20).tolist()

# test
extract_coco(inFile, outFile, subset, percentage)

inFile = f"/tempory/coco/annotations/instances_val2017.json"
outFile = f"./coco_light/coco_light_validation.json"
extract_coco(inFile, outFile, subset, percentage)
# ===============================
# COCO test (for debuggin purposes)
# ===============================
# train

inFile = f"/tempory/coco/annotations/instances_train2017.json"
outFile = f"./coco_test/coco_test_train.json"
percentage = 0.1
subset = np.random.choice(90, 4).tolist()
extract_coco(inFile, outFile, subset, percentage)

# test 

inFile = f"/tempory/coco/annotations/instances_val2017.json"
outFile = f"./coco_test/coco_test_val.json"
extract_coco(inFile, outFile, subset, percentage)
