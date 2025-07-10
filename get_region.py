import cv2

# 클릭 이벤트 처리 함수
def mouse_callback(event, x, y, flags, param):
    '''
    왼쪽 마우스 버튼을 클릭하면 그 좌표값이 나오는 함수
    '''
    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 버튼 클릭했을 때
        print(f"좌표: ({x}, {y})")

# 이미지 불러오기
img = cv2.imread('your_image.jpg')
cv2.namedWindow('image')
# 마우스 콜백 함수 등록
cv2.setMouseCallback('image', mouse_callback)

cv2.waitKey(0)
cv2.destroyAllWindows()
