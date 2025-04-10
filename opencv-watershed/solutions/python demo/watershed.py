import cv2
import numpy as np

# 读取图像
image = cv2.imread("fruits.jpg")
if image is None:
    print("Error: Unable to read the image. Please check the file path and integrity.")
else:
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 预处理：高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 生成梯度图像
    gradient = cv2.Sobel(blurred, cv2.CV_64F, 1, 1, ksize=3)

    # 将梯度图像转换为 8位无符号整数
    gradient_abs = cv2.convertScaleAbs(gradient)

    # 对梯度图像进行二值化处理
    _, binary = cv2.threshold(gradient_abs, 50, 255, cv2.THRESH_BINARY)

    # 创建种子标记
    ret, markers = cv2.connectedComponents(binary)

    # 应用分水岭算法
    markers = cv2.watershed(image, markers)

    # 显示结果
    image[markers == -1] = [0, 0, 255]  # 标记边界为红色
    cv2.imshow("Segmented Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
