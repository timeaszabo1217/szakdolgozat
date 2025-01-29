import os
import cv2


def convert_to_ycbcr(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return None
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    return ycbcr_image[:, :, 1]  # Csak a Cb (krominancia) komponens kivétele


def preprocess_images(image_dir):
    images = []
    labels = []

    # Ellenőrizzük az elérési utakat
    print(f"Checking directory: {image_dir}")
    au_tp_directories_exist = any(subdir in os.listdir(image_dir) for subdir in ['Au', 'Tp'])

    if au_tp_directories_exist:
        for subdir in ['Au', 'Tp']:
            full_dir = os.path.join(image_dir, subdir)
            print(f"Checking subdirectory: {full_dir}")
            if not os.path.exists(full_dir):
                print(f"Directory not found: {full_dir}")
                continue
            for filename in os.listdir(full_dir):
                if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.tif'):
                    file_path = os.path.join(full_dir, filename)
                    print(f"Processing Revised file: {file_path}")
                    ycbcr_image = convert_to_ycbcr(file_path)
                    if ycbcr_image is not None:
                        images.append(ycbcr_image)
                        labels.append(0 if subdir == 'Au' else 1)  # Au képek eredetiek, Tp képek hamisítottak
    else:
        for filename in os.listdir(image_dir):
            file_path = os.path.join(image_dir, filename)
            if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.tif'):
                print(f"Processing file: {file_path}")
                ycbcr_image = convert_to_ycbcr(file_path)
                if ycbcr_image is not None:
                    images.append(ycbcr_image)
                    if 'Au' in filename:
                        labels.append(0)  # Au képek eredetiek
                    elif 'Tp' in filename:
                        labels.append(1)  # Tp képek hamisítottak
                    else:
                        print(f"Could not determine label for file: {file_path}")

    if not images or not labels:
        print(f"No images or labels found in the directory: {image_dir}")
    else:
        print(f"Number of images: {len(images)}, Number of labels: {len(labels)}")

    return images, labels


if __name__ == "__main__":
    image_dir = os.path.abspath('../data/CASIA2.0_Groundtruth')
    images, labels = preprocess_images(image_dir)
    print(f"Number of loaded images: {len(images)}")
