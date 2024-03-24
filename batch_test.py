import torch
import warnings
import numpy as np
from typing import Tuple, List
from PIL import Image
import os
import time

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

def load_model():
    global model

    config_file = 'GroundingDINO/config/GroundingDINO_SwinB_cfg.py'
    checkpoint_path = 'weights/groundingdino_swinb_cogcoor.pth'
    cpu_only = False

    args = SLConfig.fromfile(config_file)
    args.device = "cuda:3" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    try:
        transform = T.Compose(
            [
                T.Resize((800, 1200)), # ? BATCHINFERENCE
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_source = Image.open(image_path).convert("RGB")
        image = np.asarray(image_source)
        image_transformed, _ = transform(image_source, None)
        return image, image_transformed
    except:
        return None

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

def predict_batch(
        model,
        images: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    caption = preprocess_caption(caption=caption)

    model = model.to(device)
    image = images.to(device)
    with torch.no_grad():
        outputs = model(image, captions=[caption for _ in range(len(images))]) # Same caption for all images
    
    prediction_logits = outputs["pred_logits"].cpu().sigmoid()  # prediction_logits.shape = (batch_size, nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()  # prediction_boxes.shape = (batch_size, nq, 4)

    batch_boxes = []
    batch_logits = []
    batch_phrases = []

    for img_logits, img_boxes in zip(prediction_logits, prediction_boxes):
        mask = img_logits.max(dim=1)[0] > box_threshold
        logits = img_logits[mask]  # logits.shape = (n, 256)
        boxes = img_boxes[mask]  # boxes.shape = (n, 4)

        tokenizer = model.tokenizer
        tokenized = tokenizer(caption)

        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit
            in logits
        ]

        batch_boxes.append(boxes.tolist())
        batch_logits.append(logits.max(dim=1)[0].tolist())
        batch_phrases.append(phrases)

    return batch_boxes, batch_logits, batch_phrases

model = load_model()
print("Model Loaded!")

caption = 'boy . girl . man . woman'
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1

image_folder_path = 'temp_storage/batch_test'
img_paths = [
    f for f in os.listdir(image_folder_path) 
    if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))  
] 
img_paths = [os.path.join(image_folder_path, path) for path in img_paths]

loaded_images = []
for img_path in img_paths:
    tmp = load_image(img_path)
    if tmp is not None:
        print(f"Adding {img_path}")
        loaded_images.append(tmp[1])
        if len(loaded_images) == BATCH_SIZE:
            break


images = torch.stack(loaded_images)

# Warmup
print(f"Warming up...")
boxes, logits, phrases = predict_batch(
    model=model,
    images=images,
    caption=caption,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
    device=DEVICE,
)

print(f"Running batch inference")
start_time = time.perf_counter()
# Perform batch inference
boxes, logits, phrases = predict_batch(
    model=model,
    images=images,
    caption=caption,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
    device=DEVICE,
)


inference_time = time.perf_counter() - start_time

print(f"Inference_time: {inference_time:.3f}s, ({inference_time / BATCH_SIZE:.3f}s/img)")

# Unpack boxes, logits, and phrases directly in the loop header for clarity
for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
    num_detection = len(box)
    print(f"Image {i + 1}")
    valid_detections = 0
    # Loop through each detection for the current image
    for j in range(num_detection):
        
        if phrase[j] != "":
            valid_detections += 1
            print(f"{phrase[j]} Bbox: {box[j]} ({logit[j]*100:.1f}%)")

    print(f"{valid_detections} detections \n")
