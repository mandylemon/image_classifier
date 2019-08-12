from functions import *
import argparse
import json
from torchvision import datasets, transforms, models

#%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description='Image Classifier Prediction')
set_predict_parser(parser)
args = parser.parse_args()

img_path = args.path_to_image
check_point_path = args.checkpoint
model = load_checkpoint(check_point_path)

if args.gpu == True :
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else :
    dev = "cpu"

probs, classes = predict(img_path, model, args.top_k, dev)

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
    
plt_data = organize_data(classes, probs, model.class_to_idx, cat_to_name)

# Create pandas DataFrame from numpy Series
df = pd.DataFrame(plt_data)

# Re-arrange and sort Dataframe 
#df = df.groupby(['names']).mean()
#df = df.sort_values('probs', ascending=False)

# Plot horizontal bar chart
#df.plot(kind = 'barh', rot=1)

print(df)
#plt.xlim(0.01, 0.02)
#plt.show()