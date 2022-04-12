# FC dataset: Preprocessing
Images are stored at data_raw.
Labels are downloaded from Labelbox. FC dataset has been annotated at Labelbox by additive manufactoring team (Margarita Guerrero). 
This process is done at the notebook: "team_work/segmentation_2020/scripts/python/analysis/ipynb/fastcool_pipepline.ipynb"

# 1. Images to data_transformed
Load images at data_raw and save them at data_transformed by **removing certaing characteres** from image names. (As " ", "(", ")", "+", "#")

# 2. Download from Labelbox
Labelbox export one image per object.
 1. Read Labelbox data from jsons stored at "data_transformed/labelbox_json".
 2. Download each labelbox object and stored them at "data_transformed/labels_curl".

# 3. Generate labels
Parse to labels where each pixel must have a value from 0 to 4:
 - 0: Austenita
 - 1: Matriz
 - 2: MA
 - 3: Precipitados
 - 4: Defectos      

The following process is done:
 1. All the objects are merged on a single image from [1 to 5] leaving as zeros the unlabeled pixels
 2. The unlabeled pixels adquire the label of its closer neightboor.
 3. Parse the labels from [1,5] to [0,4].

# 4. Generate coloured_labels
Generate coloured labels for visualizing purposes from the labels generated at the previous point. Each class has the following RGB colours:
 - 0: (128, 0, 255)
 - 1: (43, 255, 0)
 - 2: (255, 0, 0)
 - 3: (255, 0, 255)
 - 4: (255, 255, 0)

# 5. Fix labels with seg_tip predictions
Some baselines wer used with Model Assited Labeling tool at labelbox. As a result most of the labels contain noise at "Austenita" and "Matriz" labels.
As segmentation_tip unsupervised methods generates most accurate labels, we generate new labels by doing a merge between seg_tip predictions and the original labels.
We load "Matriz" and "Austenita" annotations from the seg_tip predictions and "paste" the annotations of the rest of the labels from the original labels.