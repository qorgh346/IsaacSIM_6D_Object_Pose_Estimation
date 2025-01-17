import cv2
import os
import time

def display_images(image_folder, delay=0.5):
    """
    Display 2D images (RGB, Mask, Depth) from a single folder one after another.

    :param image_folder: Path to the folder containing images.
    :param delay: Delay in seconds between each image frame. Default is 0.5 seconds.
    """
    # Get the list of image files in the folder
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

    # Filter image files by type
    rgb_files = [f for f in image_files if "_color" in f]
    mask_files = [f for f in image_files if "_mask" in f]
    depth_files = [f for f in image_files if "_depth" in f]

    # Ensure that all types have the same number of images
    assert len(rgb_files) == len(mask_files) == len(depth_files), "All types must have the same number of images."
    resize_factor = 0.5
    # Iterate through the image files and display them
    i = 0
    paused = False
    while i < len(rgb_files):
        if not paused:
            # Read the images using OpenCV
            rgb_image = cv2.imread(os.path.join(image_folder, rgb_files[i]))
            mask_image = cv2.imread(os.path.join(image_folder, mask_files[i]))
            depth_image = cv2.imread(os.path.join(image_folder, depth_files[i]))

            # Resize the images
            rgb_image_resized = cv2.resize(rgb_image, (0, 0), fx=resize_factor, fy=resize_factor)
            mask_image_resized = cv2.resize(mask_image, (0, 0), fx=resize_factor, fy=resize_factor)
            depth_image_resized = cv2.resize(depth_image, (0, 0), fx=resize_factor, fy=resize_factor)

            # Show the images
            cv2.imshow('RGB Image Sequence', rgb_image_resized)
            cv2.imshow('Mask Image Sequence', mask_image_resized)
            cv2.imshow('Depth Image Sequence', depth_image_resized)

            i += 1

        # Wait for the specified delay in milliseconds
        key = cv2.waitKey(int(delay * 1000))

        # If the user presses 'q', exit the loop
        if key == ord('q'):
            break

        # If the user presses 'w', pause or continue the playback
        if key == ord('w'):
            paused = not paused

    # Close the windows and release resources
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image_folder = '/media/ailab/새 볼륨/hojun_ws/data/IsaacSIM/test_0425/Viewport_1'
    display_images(image_folder, delay=0.5)