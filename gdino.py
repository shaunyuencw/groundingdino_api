import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

import time
from tqdm import tqdm
import warnings

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    print(boxes)
    print(labels)
    assert len(boxes) == len(labels), "Boxes and labels must have the same length."

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    for box, label in zip(boxes, labels):
        box = box * torch.Tensor([W, H, W, H])
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        color = tuple(np.random.randint(0, 255, size=3).tolist())

        x0, y0, x1, y1 = box.int().tolist()

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)

        try:
            font_size = 20
            font = ImageFont.truetype("fonts/Arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
            print("Custom font not found, using default.")

        # Calculate text size
        text_width = draw.textlength(str(label), font=font)
        font_height = font_size
        
        # Top left corner of the bounding box
        text_x = x0
        text_y = y0 - font_height - 4  # Adjust as needed to move text above the box

        # Ensure text background does not go out of image bounds
        if text_y < 0:
            text_y = y0

        # Draw text background and text
        text_bbox = [text_x, text_y, text_x + text_width, text_y + font_height]
        draw.rectangle(text_bbox, fill=color)
        draw.text((text_x, text_y), str(label), fill="white", font=font)

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask



def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


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

class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def infer_and_visualize(image_path, text_prompt, box_threshold=0.3, text_threshold=0.3, save=False, show=False):

    config_file = 'GroundingDINO/config/GroundingDINO_SwinB_cfg.py'
    checkpoint_path = 'weights/groundingdino_swinb_cogcoor.pth'
    output_dir= 'infer'
    token_spans=None

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    image_pil, image = load_image(image_path)

    # Load model
    with SuppressPrints():
        model = load_model(config_file, checkpoint_path)

    # Start the timer
    start_time = time.perf_counter()

    # Run model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold,
            cpu_only=False, token_spans=token_spans
        )

    # End the timer and calculate elapsed time
    inference_time = time.perf_counter() - start_time

    # Prepare data for visualization
    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # Height, Width
        "labels": pred_phrases,
    }

    image_with_box = plot_boxes_to_image(image_pil.copy(), pred_dict)[0]

    if save:
        image_with_box.save(os.path.join(output_dir, "pred.jpg"))

    if show:
        print(f"Inference Time: {inference_time:.2f} seconds.")
        
        # Display raw vs pred
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
        # Displaying the raw image
        ax[0].imshow(image_pil)
        ax[0].axis('off')
        ax[0].set_title('Raw Image')
    
        # Displaying the image with boxes
        ax[1].imshow(image_with_box)
        ax[1].axis('off')
        ax[1].set_title('Prediction Image')
    
        plt.tight_layout()
        plt.show()

    return inference_time

