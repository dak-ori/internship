import cv2
import numpy as np
from ultralytics import YOLO
import json
import os

def load_image(image_path):
    """이미지를 원본 크기로 로드"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {image_path}")
    h, w = image.shape[:2]
    return image, (w, h)  # (width, height)

def predict_segmentation(model, image, classes=[0], imgsz=640, conf=0.4):
    """YOLO 모델 세그멘테이션 예측"""
    results = model.predict(image, classes=classes, imgsz=imgsz,conf=conf)
    return results

def extract_polygon_coordinates(results):
    """예측 결과에서 폴리곤 좌표 추출"""
    masks = results[0].masks
    polygon_coords = []
    if masks is not None and masks.xy is not None:
        for poly in masks.xy:
            pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
            polygon_coords.append(pts.tolist())
    return polygon_coords

def draw_polygons(image, polygon_coords, color=(0, 255, 0), thickness=2):
    """이미지에 폴리곤 그리기"""
    for poly in polygon_coords:
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)
    return image

def save_polygons_to_json(polygon_coords, json_path, label="person", base_size=(1120, 746)):
    """비율 기반으로 JSON 저장"""
    base_w, base_h = base_size
    polygon_data = []
    for poly in polygon_coords:
        coord_str = ",".join(
            f"{x / base_w:.6f},{y / base_h:.6f}" for x, y in poly
        )
        polygon_data.append({
            "label": label,
            "coords": coord_str,
            "shape": "poly"
        })

    with open(json_path, 'w') as f:
        json.dump(polygon_data, f, indent=2)

def main(image_path):
    model = YOLO("models/yolo11m-seg.pt")
    image, base_size = load_image(image_path)
    results = predict_segmentation(model, image, classes=[0])
    polygon_coords = extract_polygon_coordinates(results)
    image_with_polygons = draw_polygons(image, polygon_coords)

    filename = os.path.splitext(os.path.basename(image_path))[0]
    save_polygons_to_json(polygon_coords, f"{filename}.json", base_size=base_size)

if __name__ == "__main__":
    main("static/kimbap.png")
