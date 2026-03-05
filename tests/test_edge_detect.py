import cv2
import numpy as np
import os


def rgb_to_canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 80, 180)

    # Optional: thicken edges slightly (better for SD guidance)
    edges = cv2.dilate(edges, None)

    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def rgb_to_sobel(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))

    return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)


def rgb_to_lineart(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)

    lineart = cv2.divide(gray, 255 - blurred, scale=256)

    return cv2.cvtColor(lineart, cv2.COLOR_GRAY2BGR)


def process_image(input_path):
    if not os.path.exists(input_path):
        print("Input file not found.")
        return

    image = cv2.imread(input_path)

    if image is None:
        print("Failed to load image.")
        return

    # Generate edges
    canny_edges = rgb_to_canny(image)
    sobel_edges = rgb_to_sobel(image)
    lineart_edges = rgb_to_lineart(image)

    base_name = os.path.splitext(input_path)[0]

    cv2.imwrite(base_name + "_canny.png", canny_edges)
    cv2.imwrite(base_name + "_sobel.png", sobel_edges)
    cv2.imwrite(base_name + "_lineart.png", lineart_edges)

    print("Edge images saved:")
    print(base_name + "_canny.png")
    print(base_name + "_sobel.png")
    print(base_name + "_lineart.png")


if __name__ == "__main__":
    input_files = ["./assets/quiz-image.png", "./assets/riddle-image.png"]
    for input_file in input_files:
        process_image(input_file)