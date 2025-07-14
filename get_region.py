import cv2

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"좌표: ({x}, {y})")

img = cv2.imread('static/merry.png')
if img is None:
    print("Check Image_Path.")
    exit()

cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
    merry.jpg
    사람 있는 부분 [185, 185, 535, 370]
'''
'''
    1번째 남자 [0, 430, 304, 1367]
    2번째 여자 [304,550,520,1367]
    3번째 여자 [520,550,800,1367]
    4번째 남자 [800,480,1080,1367]
    5번째 여자 [1080,565,1340,1367]
    나머지 부분은 전체 이미지맵으로 하자
'''
'''
    소주병 [1260,630,1310,850]
'''