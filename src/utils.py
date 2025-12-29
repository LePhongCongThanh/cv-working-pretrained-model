import cv2
import matplotlib.pyplot as plt

def plot_image(image_path):
    """
    Helper to plot an image using matplotlib.
    """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

def count_classes(labels_path):
    """
    Count class occurrences in a label file or directory of labels.
    """
    # Placeholder for utility logic
    pass
