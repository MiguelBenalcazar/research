from datasets import load_dataset
import pickle
from pycocotools import mask
import matplotlib.pyplot as plt

from pycocotools import mask as coco_mask

from skimage.measure import regionprops, label
import numpy as np
from PIL import Image

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("nvidia/describe-anything-dataset", "COCOStuff")


def main():
    data =ds["train"]
    enum = enumerate(data)
    item = data[5]#next(enum)
    # keys = item[1].keys()

  
    print(item)

    img = item['jpg']
    img.show()

    pickle_raw_data = item['pickle']
    data = pickle.loads(pickle_raw_data)

    print(data)

    mask_rle = data[0]['mask_rle']

    # Decode the RLE to a binary mask
    decoded_mask = mask.decode(mask_rle)  # numpy array of shape (H, W)
    # Convert image to numpy
    # img_np = np.array(img)

    # # Ensure mask has same shape as image height/width
    # if decoded_mask.shape != img_np.shape[:2]:
    #     decoded_mask = Image.fromarray(decoded_mask).resize(img_np.shape[:2][::-1], resample=Image.NEAREST)
    #     decoded_mask = np.array(decoded_mask)

    # # Apply mask (zero out background)
    # img_np[decoded_mask == 0] = 0  # black out non-masked areas

    # # Convert back to PIL for viewing/saving
    # masked_image = Image.fromarray(img_np)
    # masked_image.show()

    # # Convert binary mask to regions
    # label_mask = label(decoded_mask)
    # props = regionprops(label_mask)

    # descriptions = []
    # for i, region in enumerate(props):
    #     area = region.area
    #     centroid = region.centroid
    #     bbox = region.bbox
    #     descriptions.append(
    #         f"Region {i+1} has area {area} pixels, centroid at {centroid}, bounding box {bbox}."
    #     )

    # text_prompt = " ".join(descriptions)


    # print(text_prompt)

    # Visualize the mask
    plt.imshow(decoded_mask, cmap='gray')
    plt.title("Decoded Segmentation Mask")
    plt.axis('off')
    plt.show()


    # keys = data[0].keys()
    # for i in keys:
    #     print(i)
    #     print(data[0][i], "\n")




if __name__ == "__main__":
    main()