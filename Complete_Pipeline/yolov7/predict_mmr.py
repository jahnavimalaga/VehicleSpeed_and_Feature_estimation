import argparse
import torch
from torchvision import transforms
import pandas as pd
import os

from utils.train_mmr import initialize_model

from PIL import Image

import torch.nn.functional as F




device = torch.device("cpu")


def predict(img, model_name="resnet50_100epochs.pt", k=5):

    img_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    img = Image.open(file)
    if img.mode != "RGB":  # Convert png to jpg
        img = img.convert("RGB")
    img = img_transforms(img)
    img = img.to(device)
    img = img.unsqueeze(0)  # Add batch dimension (because single image)
    print(img.size())

    df = pd.read_pickle("data/preprocessed_data_mmr.pkl")
    num_classes = df["Classname"].nunique()
    print("model name:",model_name[:8])
    model, _ = initialize_model(model_name[:8], num_classes, feature_extract=True)
    path = os.path.dirname(__file__)
    model.load_state_dict(torch.load(path + "models/" + str(model_name), map_location=device))
    model.to(device)
    model.eval()

    pd.set_option('display.max_rows', None)

    with torch.no_grad():
        output = model(img)
        _, preds = torch.topk(output, k)

    preds = torch.transpose(preds, 0, 1)
    preds = preds.cpu()  # Send tensor to cpu
    preds = pd.DataFrame(preds.numpy(), columns=["Classencoded"])  # Convert to dataframe

    class_encoded_matches = pd.merge(df, preds, how="inner")
    class_encoded_matches = pd.merge(preds, class_encoded_matches, how="left", on="Classencoded", sort=False)  # Preserves ordering
    classname_matches = class_encoded_matches["Classname"].unique()

    return classname_matches



def mmr_predict(img, model, img_transforms,device,df,k=5):
    
    img = img_transforms(img)
    img = img.to(device)
    img = img.unsqueeze(0)  # Add batch dimension (because single image)
    #print(img.size())

    pd.set_option('display.max_rows', None)

    with torch.no_grad():
        output = model(img)
        # Printing the loss value
        #eprint("shape of output:",output.shape)
        output = F.softmax(output, dim=1)

        probs, preds = torch.topk(output, k)
        #print(f"output:{temp},preds:{preds}")
    
    #probs = torch.transpose(probs, 0, 1)
    probs = torch.round(probs * 100) / 100
    probs = probs.cpu().numpy()  # Send tensor to cpu

    preds = torch.transpose(preds, 0, 1)
    preds = preds.cpu()  # Send tensor to cpu
    #print("probs:",probs)
    #print("preds:",preds.numpy())
    preds = pd.DataFrame(preds.numpy(), columns=["Classencoded"])  # Convert to dataframe

    class_encoded_matches = pd.merge(df, preds, how="inner")
    class_encoded_matches = pd.merge(preds, class_encoded_matches, how="left", on="Classencoded", sort=False)  # Preserves ordering
    classname_matches = class_encoded_matches["Classname"].unique()

    #print(f"classname:{classname_matches},probs:{probs[0]}")
    combined_results = list(zip(classname_matches,probs[0]))
    return combined_results#classname_matches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="path to image")
    parser.add_argument("-m", "--model", type=str, help="model name")
    parser.add_argument("-k", "--topk", type=int, help="top k predictions")
    args = vars(parser.parse_args())

    path = args["path"]
    model_name = args["model"]
    k = args["topk"]

    #classname_matches = predict(path, model_name, k)
    #print(classname_matches)


    img = Image.open(path)
    if img.mode != "RGB":  # Convert png to jpg
        img = img.convert("RGB")
    img_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    df = pd.read_pickle("data/preprocessed_data_mmr.pkl")
    num_classes = df["Classname"].nunique()
    print("model name:",model_name[:8])
    model, _ = initialize_model(model_name[:8], num_classes, feature_extract=True)
    path = os.path.dirname(__file__)
    model.load_state_dict(torch.load(path + "models/" + str(model_name), map_location=device))
    model.to(device)
    model.eval()
    classname_matches = mmr_predict(img, model, img_transforms,device,df,k)
    print(classname_matches)

if __name__ == "__main__":
    main()
#python predict_mmr.py -p ../../test/0_0_1_car_0.96.jpg -m resnet50_40epochs_mmr.pt -k 3