import cv2

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"좌표: ({x}, {y})")

img = cv2.imread('static/bottle.png')
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
    1번째 남자 [0,213,130,780]
    2번째 여자 [150,265,265,790]
    3번째 여자 [275,270,400,805]
    4번째 남자 [405,235,510,600]
    5번째 여자 [540,300,669,755]
    윗공간 여백 [0,0,669,221]
'''
'''
    소주병 [1260,630,1310,850]
'''