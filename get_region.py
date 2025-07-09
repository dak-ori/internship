import cv2
import numpy as np
from ultralytics import YOLO
import json

def load_and_resize_image(image_path, size=(1000, 1481)):
    """이미지 로드 후 리사이즈"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {image_path}")
    resized_image = cv2.resize(image, size)
    return resized_image

def predict_segmentation(model, image, classes=[0], imgsz=640):
    """YOLO 모델로 이미지 세그멘테이션 예측"""
    results = model.predict(image, classes=classes, imgsz=imgsz, conf=0.4)
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

def save_polygons_to_json(polygon_coords, json_path, label="person"):
    """HTML 이미지맵에서 읽을 수 있게 coords를 문자열로 저장"""
    polygon_data = []
    for poly in polygon_coords:
        coord_str = ",".join(f"{x},{y}" for x, y in poly)
        polygon_data.append({
            "label": label,
            "coords": coord_str,
            "shape": "poly"
        })
    
    with open(json_path, 'w') as f:
        json.dump(polygon_data, f, indent=2)


def main(image_path):
    model = YOLO("models/yolo11n-seg.pt")
    image = load_and_resize_image(image_path)
    results = predict_segmentation(model, image, classes=[0], imgsz=640)
    polygon_coords = extract_polygon_coordinates(results)
    image_with_polygons = draw_polygons(image, polygon_coords)
    save_polygons_to_json(polygon_coords, "polygons.json")
    print(model.names)

    # 폴리곤 좌표 출력
    # for i, coords in enumerate(polygon_coords):
    #     print(f"Polygon {i} coordinates:")
    #     for x, y in coords:
    #         print(f"({x}, {y})") # 1차원 리스트 출력
    #     print()
    
    # for i, coords in enumerate(polygon_coords):
    #     print(f"Polygon {i} coordinates:")
    #     print(coords)  # 2차원 리스트 전체 출력
    #     print()
        
    # 결과 보여주기
    cv2.imshow("Accurate Polygon", image_with_polygons)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main("static/net.jpg")
