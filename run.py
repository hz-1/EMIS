import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 
import os
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import argparse
from utils.compute import get_centroids_and_bounding_boxes
from PIL import Image
from efficientvit.efficientvit.sam_model_zoo import create_sam_model
from efficientvit.efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
import json

"""Max Entropy Point"""
def image_entropy(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate the histogram
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    # Normalize the histogram
    hist /= hist.sum()
    # Calculate the entropy
    entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))

    return entropy

def calculate_image_entroph(img1, img2):
    # Calculate the entropy for each image
    entropy1 = image_entropy(img1)
    # print(img2)
    try:
        entropy2 = image_entropy(img2)
    except:
        entropy2 = 0
    # Compute the entropy between the two images
    entropy_diff = abs(entropy1 - entropy2)
    # print("Entropy Difference:", entropy_diff)
    return entropy_diff

def select_grid(image, center_point, grid_size):
    (img_h, img_w, _) = image.shape

    # Extract the coordinates of the center point
    x, y = center_point
    x = int(np.floor(x))
    y = int(np.floor(y))
    # Calculate the top-left corner coordinates of the grid
    top_left_x = x - (grid_size // 2) if x - (grid_size // 2) > 0 else 0
    top_left_y = y - (grid_size // 2) if y - (grid_size // 2) > 0 else 0
    bottom_right_x = top_left_x + grid_size if top_left_x + grid_size < img_w else img_w
    bottom_right_y = top_left_y + grid_size if top_left_y + grid_size < img_h else img_h

    # Extract the grid from the image
    grid = image[top_left_y: bottom_right_y, top_left_x: bottom_right_x]

    return grid

def get_entropy_points(input_point,mask,image):
    max_entropy_point = [0,0]
    max_entropy = 0
    grid_size = 9
    center_grid = select_grid(image, input_point, grid_size)

    indices = np.argwhere(mask ==True)
    for x,y in indices:
        grid = select_grid(image, [x,y], grid_size)
        entropy_diff = calculate_image_entroph(center_grid, grid)
        if entropy_diff > max_entropy:
            max_entropy_point = [x,y]
            max_entropy = entropy_diff
    return [max_entropy_point[1], max_entropy_point[0]]

def get_embedding(img, predictor):
    predictor.set_image(img)
    # img_embedding = predictor.get_image_embedding()
    img_embedding = predictor.features
    return img_embedding

def train(args, predictor):
    data_path = args.train_path
    image_embeddings = []
    labels = []
    filenames = ['00000.jpg', '00001.jpg', '00002.jpg', '00003.jpg', '00004.jpg']
    # print(filenames)
    i = 0 
    for filename in tqdm(filenames):
        image = cv2.imread(os.path.join(data_path, 'images', filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(data_path, 'masks', filename))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY) 
        downsampled_mask = cv2.resize(mask, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        img_embedding = get_embedding(image, predictor)
        img_embedding = img_embedding.cpu().numpy().transpose((2, 3, 1, 0)).reshape((64, 64, 256)).reshape(-1, 256)
        image_embeddings.append(img_embedding)
        labels.append(downsampled_mask.flatten())
        i += 1
        if i > args.images: break

    image_embeddings_cat = np.concatenate(image_embeddings)
    labels = np.concatenate(labels)
    model = LogisticRegression(max_iter=1000)
    model.fit(image_embeddings_cat, labels)

    return model

def test(args, model, predictor):
    data_path = args.input_path
    filenames = os.listdir(data_path)

    for filename in tqdm(filenames):

        img_path = os.path.join(data_path, filename)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_height, orig_width = image.shape[:2]

        with Image.open(img_path) as input_image:
            reshaped_image = input_image.resize((1000, 1000), resample=Image.BICUBIC)
            reshaped_image_np = np.array(reshaped_image)

        image = reshaped_image_np  
        img = reshaped_image_np    

        # img_path = os.path.join(data_path, filename)
        # image = cv2.imread(img_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # orig_height, orig_width = image.shape[:2]
        
        # input_image = Image.open(img_path)
        # reshaped_image = input_image.resize((1000, 1000), resample=Image.BICUBIC)
        # reshaped_image.save(img_path)

        # image = cv2.imread(img_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_embedding = get_embedding(image, predictor)
        img_embedding = img_embedding.cpu().numpy().transpose((2, 3, 1, 0)).reshape((64, 64, 256)).reshape(-1, 256)
        # print(img_embedding.shape)
        y_pred = model.predict(img_embedding)
        # print(y_pred.shape)
        y_pred = y_pred.reshape((64, 64))  
        
        mask_pred_l = cv2.resize(y_pred, (1000, 1000), interpolation=cv2.INTER_CUBIC)
        kernel = np.ones((5, 5), np.uint8)
        eroded_mask = cv2.erode(mask_pred_l, kernel, iterations=3)
        mask_pred_l = cv2.dilate(eroded_mask, kernel, iterations=5)
        
        points,boxs = get_centroids_and_bounding_boxes(mask_pred_l)
        # print(len(points), len(boxs))
        mask_matrices1 = []
        # mask_matrices2 = []
        id = 0 
        mask_image = Image.fromarray(mask_pred_l * 255, mode='L')
        mask_image = mask_image.resize((orig_width, orig_height), resample=Image.NEAREST)
        # mask_image.save(os.path.join(args.save_path, filename.split('.')[0])+'_corse.png')

        for point, box  in zip(points, boxs):
            fg_point = point
            input_point = np.array([fg_point])
            input_label = np.array([1])

            predictor.set_image(image)
            input_point = np.array([fg_point])
            input_box = np.array(box)
            input_label = np.array([1])
            masks1, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_box,
                multimask_output=False,
            ) 
            mask1 = masks1[0].astype(int)
            mask_matrices1.append(mask1)
            '''
            #prompt the sam with the Aug_point
            im = np.asarray(Image.open(os.path.join(data_path, 'images', filename)).convert('RGB'))
            predictor.set_image(im)
            entropy_point = get_entropy_points(fg_point, mask_p, im)
            input_point = np.array([fg_point, entropy_point])
            input_label = np.array([1, 1])
             masks2, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_box,
                multimask_output=False,
            ) 
            mask2 = masks2[0].astype(int)
            mask_matrices2.append(mask2)
            show_mask(mask2, plt.gca())
            show_points(input_point, input_label, plt.gca())
            '''
            '''
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 255), thickness=3)
            cv2.circle(img, (int(point[0]), int(point[1])), 1, (0, 255, 0), 3)
            id = id + 1
        cv2.imwrite(os.path.join(args.save_path, "vis_{}.png".format(filename.split('.')[0])), img)'''

        height, width = mask_matrices1[0].shape
        result_image = Image.new('L', (width, height), color=0)

        for mask in mask_matrices1:
            for y in range(height):
                for x in range(width):
                    if mask[y][x] == 1:
                        result_image.putpixel((x, y), 255)

        w, h = result_image.size
        pixel_values = list(result_image.getdata())
        result_mask1 = np.array(pixel_values).reshape(h, w)
        
        result_image = result_image.resize((orig_width, orig_height), resample=Image.BICUBIC)
        result_image.save(os.path.join(args.save_path, "{}.png".format(filename.split('.')[0])))
    
        input_image = Image.open(img_path)
        reshaped_image = input_image.resize((orig_width, orig_height), resample=Image.BICUBIC)
        reshaped_image.save(img_path)

def output_anchors(images_folder, json_folder):

    for file_name in os.listdir(images_folder):
        if file_name.endswith('.png'):
            png_path = os.path.join(images_folder, file_name)
            image = cv2.imread(png_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            anchors = []

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                anchors.append((x, y, w, h))
            json_data = {
                'anchors': anchors
            }
            base_name = os.path.splitext(file_name)[0]
            json_file_path = os.path.join(json_folder, f"{base_name}.json")
            with open(json_file_path, 'w') as json_file:
                json.dump(json_data, json_file, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--images', type=int, default=5, help='num of traing images')
    parser.add_argument('--train_path', type=str, default='./few_shot', help='path to train data')
    parser.add_argument('--input_path', type=str, default='./your_data', help='path to input data')
    parser.add_argument('--model_type', type=str, default='vit_h', help='SAM model type')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/sam_vit_h_4b8939.pth', help='SAM checkpoint')
    parser.add_argument('--save_path', type=str, default='./output/img', help='path to the png results')
    parser.add_argument('--json_path', type=str, default='./output/json', help='path to the json results')
    args = parser.parse_args()

    efficientvit_sam = create_sam_model(
    name="xl0", weight_url= "efficientvit/assets/checkpoints/sam/xl0.pt"
    )
    efficientvit_sam = efficientvit_sam.cuda().eval()
    efficientvit_sam_predictor = EfficientViTSamPredictor(efficientvit_sam)

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)
    global predictor
    # predictor = SamPredictor(sam)
    predictor = efficientvit_sam_predictor
    print(args.device)
    print('SAM model loaded!', '\n')
    model = train(args, predictor)
    test(args, model, predictor)
    output_anchors(args.save_path, args.json_path)


if __name__ == '__main__':
    main()
