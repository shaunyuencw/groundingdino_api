from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from typing import List
import base64
import os
from io import BytesIO
from PIL import Image
import time
import string
import warnings

import torch
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
def load_model():
    global model

    config_file = 'GroundingDINO/config/GroundingDINO_SwinB_cfg.py'
    checkpoint_path = 'weights/groundingdino_swinb_cogcoor.pth'
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
            <title>GroundingDINO Demo</title>
        </head>
        <body>
            <h1>GroundingDINO Demo Updated</h1>
        </body>
    </html>
    """

@app.post("/detect/")
async def detect_objects(image: UploadFile = File(...), text_prompt: str = Form(...), 
                         box_threshold: float = Form(0.3), text_threshold: float = Form(0.3)):
    global model
    gpu_id = os.getenv("GPU_ID", "Not set")
    print(f"Processing request on GPU: {gpu_id}")

    print(f"Verifying params: text_prompt ({text_prompt}), box_threshold ({box_threshold}), text_threshold ({text_threshold})")

    # TEMP_IMAGES_DIR = "temp_images"
    # os.makedirs(TEMP_IMAGES_DIR, exist_ok=True)

    try:
        # Read image as bytes
        image_data = await image.read()
        # Open the image with PIL
        image_pil = Image.open(BytesIO(image_data)).convert("RGB")

        # # Create a timestamped filename
        # timestamp = int(time.time())
        # temp_image_filename = f"{timestamp}_{image.filename.rsplit('.', 1)[0]}.jpg"  # Default to .jpg if no extension found
        # temp_image_path = os.path.join(TEMP_IMAGES_DIR, temp_image_filename)
        
        # # Save the image for verification, specify format as JPEG
        # image_pil.save(temp_image_path, 'JPEG')
        # print(f"Image saved for verification at: {temp_image_path}")

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Preprocess the image
    _, image_tensor = load_image(image_pil)

    text_prompt = text_prompt.lower() # Convert to lower case
    
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
            # print(f"l: {label} detected with {conf}% confidence")

        # else:
        #     print(f"?: {label} detected with {conf}% confidence")
        #     print(f"{label} might be inrelevant...")
    return {"bounding_boxes": valid_boxes, "labels": labels_list, "confidence": conf_list, "inference_time": inference_time}