# Standard library imports
import base64
import os
import time
import string
import warnings
from io import BytesIO
from typing import List
import datetime

# Related third party imports
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image
import torch

# Local application/library specific imports
from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

warnings.filterwarnings('ignore', category=FutureWarning, message=".*The `device` argument is deprecated.*")
warnings.filterwarnings('ignore', category=UserWarning, message=".*please pass in use_reentrant=True or use_reentrant=False explicitly.*")
warnings.filterwarnings('ignore', category=UserWarning, message=".*None of the inputs have requires_grad=True.*")
warnings.filterwarnings('ignore', category=UserWarning, message=".*torch.meshgrid: in an upcoming release.*")
warnings.filterwarnings('ignore', category=UserWarning, message=".*_IncompatibleKeys.*")

model = None
device = "cuda"

def load_model():
    global model

    # config_file = 'GroundingDINO/config/GroundingDINO_SwinB_cfg.py'
    # checkpoint_path = 'weights/groundingdino_swinb_cogcoor.pth'

    config_file = 'GroundingDINO/config/GroundingDINO_SwinT_OGC.py'
    checkpoint_path = 'weights/groundingdino_swint_ogc.pth'


    cpu_only = False

    args = SLConfig.fromfile(config_file)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def load_image(image_pil):
    # Assuming the transformation and normalization steps are the same,
    # directly start with those steps without loading the image from the path
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases

    return boxes_filt, pred_phrases

app = FastAPI()

# Set up CORS middleware options
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://192.168.1.51:3000",
        "*"
        ], # Whitelist
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

@app.on_event("startup")
async def startup_event():
    load_model() # Load the model on startup
    print("Model loaded, ready to receive requests! :)")

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>G-DINO Slave</title>
        </head>
        <body>
            <h1>GroundingDINO Demo Slave</h1>
        </body>
    </html>
    """

@app.post("/detect")
async def detect_objects(request: Request):
    global model
    try:
        request_data = await request.json()

        # Extract the required variables from the request data
        # image_tensor = torch.tensor(request_data["image"], device=device)

        # Extract the required variables from the request data
        image_data = base64.b64decode(request_data["image"].encode('utf-8'))
        text_prompt = request_data["text_prompt"]
        box_threshold = request_data["box_threshold"]
        text_threshold = request_data["text_threshold"]

        image = Image.open(BytesIO(image_data))
        # Open the image with PIL
        _, image_tensor = load_image(image)

    except Exception as e:
        print(f"Error decoding JSON: {e}")
        return {"error": str(e)}

    # Perform inference
    start_time = time.perf_counter()
    boxes, labels = get_grounding_output(model, image_tensor, text_prompt, box_threshold, text_threshold)
    inference_time = f"{time.perf_counter() - start_time:.3f}"
    
    # Convert boxes and labels to a serializable format
    boxes_list = boxes.tolist()
    pred_list = [str(label) for label in labels]
    

    # Initialize empty lists for labels and confidences
    valid_boxes = []
    labels_list = []
    conf_list = []

    # Iterate over each item in the prediction list
    for i, item in enumerate(pred_list):
        # Split the item into label and confidence
        label, conf = item.rsplit('(', 1)
        conf = conf.rstrip(')')  # Remove the trailing ')' from the confidence
        label = label.replace('. [SEP]' , '')

        # Remove all punctuation and symbols from label
        label = label.translate(str.maketrans('', '', string.punctuation))

        if label.replace(' ', '') == "":
            print(f"Skipping invalid label, label ({label}), confidence {conf}")
            continue

        elif text_prompt in label.lower():
            labels_list.append(text_prompt)  # Append the label to labels_list
            conf_list.append(float(conf))  # Convert the confidence to float and append to conf_list
            valid_boxes.append(boxes_list[i])
            # print(f"tp: {text_prompt} detected with {conf}% confidence")

        elif label.lower() in text_prompt:
            labels_list.append(label)
            conf_list.append(float(conf))
            valid_boxes.append(boxes_list[i])

    return {"bounding_boxes": valid_boxes, "labels": labels_list, "confidence": conf_list, "inference_time": inference_time}