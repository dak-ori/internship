import cv2
import numpy as np
import json
import os
from fastsam import FastSAM, FastSAMPrompt

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {image_path}")
    return image

def get_mask_coords(mask):
    """마스크의 외곽선을 찾아서 polygon 좌표 추출"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    return largest.squeeze().tolist()  # (x, y)

def normalize_coords(coords, base_size):
    w, h = base_size
    return [(x / w, y / h) for x, y in coords]

def save_polygons_to_json(polygon_coords, json_path, label="person", base_size=(1000, 1481)):
    base_w, base_h = base_size
    polygon_data = []
    for poly in polygon_coords:
        coord_str = ",".join(f"{x:.6f},{y:.6f}" for x, y in normalize_coords(poly, (base_w, base_h)))
        polygon_data.append({
            "label": label,
            "coords": coord_str,
            "shape": "poly"
        })

    with open(json_path, 'w') as f:
        json.dump(polygon_data, f, indent=2)

def main(image_path, model_path='FastSAM-x.pt'):
    model = FastSAM(model_path)
    image = load_image(image_path)

    everything_results = model(
        image,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        retina_masks=True,
        imgsz=1024,
        conf=0.4,
        iou=0.9,
    )

    prompt_process = FastSAMPrompt(image, everything_results, device='cuda')
    masks = prompt_process.everything_prompt()

    polygon_coords = []
    for m in masks:
        binary_mask = (m.astype(np.uint8)) * 255
        coords = get_mask_coords(binary_mask)
        if len(coords) >= 6:
            polygon_coords.append(coords)

    filename = os.path.splitext(os.path.basename(image_path))[0]
    save_polygons_to_json(polygon_coords, f"{filename}.json", base_size=image.shape[1::-1])

    print(f"저장 완료: {filename}.json")

if __name__ == "__main__":
    main("static/net.jpg")
