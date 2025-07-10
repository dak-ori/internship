import cv2

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"좌표: ({x}, {y})")

img = cv2.imread('static/net.jpg')
if img is None:
    print("Check Image_Path.")
    exit()

cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
넷플릭스 시리즈
1 : 400, 1120
2 : 400, 1160
3 : 600, 1160
4 : 600, 1120
'''

'''
NetFlix
1 : 320,1355
2 : 320,1420
3 : 540,1420
4 : 540,1355 
'''

'''
폭삭 속았수다
1 : 180, 1175
2 : 180, 1325
3 : 820, 1325
4 : 820, 1175
'''