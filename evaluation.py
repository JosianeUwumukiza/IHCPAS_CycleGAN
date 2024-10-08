import numpy as np
from skimage.metrics import structural_similarity as ssim
from glob import glob
import argparse
import torch
from collections import OrderedDict  # Add this line
from models import create_model
from options.test_options import TestOptions
from data import create_dataset
import cv2

def load_data(image_paths, img_size):
    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)
        img = img / 255.0
        images.append(img)
    return np.array(images)

def calculate_metrics(real_images, generated_images):
    # Calculate SSIM
    ssim_scores = []
    for real_img, gen_img in zip(real_images, generated_images):
        ssim_score = ssim(real_img, gen_img, multichannel=True)
        ssim_scores.append(ssim_score)
    avg_ssim = np.mean(ssim_scores)

    # Calculate FID - This is a placeholder, you need a proper implementation
    fid_score = np.random.random()  # Replace with actual FID calculation

    # Calculate Inception Score - This is a placeholder, you need a proper implementation
    inception_score = np.random.random()  # Replace with actual Inception Score calculation

    return avg_ssim, fid_score, inception_score

def load_model_state(model, state_dict_path):
    state_dict = torch.load(state_dict_path, map_location='cpu')
    # Ensure the keys match the expected format
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if not k.startswith('module.'):
            name = 'module.' + k  # Add 'module.' prefix if not present
        else:
            name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate CycleGAN model')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--name', required=True, help='name of the experiment')
    parser.add_argument('--model', required=True, help='model type')
    opt = parser.parse_args()

    # Load options for CycleGAN
    test_opt = TestOptions().parse()  # get test options
    test_opt.dataroot = opt.dataroot
    test_opt.name = opt.name
    test_opt.model = opt.model
    test_opt.num_threads = 0   # test code only supports num_threads = 0
    test_opt.batch_size = 1    # test code only supports batch_size = 1
    test_opt.serial_batches = True  # disable data shuffling
    test_opt.no_flip = True    # no flip
    test_opt.display_id = -1   # no visdom display

    # Create dataset and model
    dataset = create_dataset(test_opt)
    model = create_model(test_opt)
    model.setup(test_opt)

    img_size = (256, 256)

    # Load the test images
    test_image_paths_A = sorted(glob(f"{opt.dataroot}/testA/*"))
    test_image_paths_B = sorted(glob(f"{opt.dataroot}/testB/*"))

    real_images_A = load_data(test_image_paths_A, img_size)
    real_images_B = load_data(test_image_paths_B, img_size)

    # Load saved models
    load_model_state(model.netG_A, 'final_A2B_net_G_A.pth')
    load_model_state(model.netG_B, 'final_B2A_net_G_B.pth')

    # Generate images
    model.netG_A.eval()
    model.netG_B.eval()

    with torch.no_grad():
        generated_images_B = []
        for img in real_images_A:
            img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).unsqueeze(0).float().to(model.device)
            gen_img_tensor = model.netG_A(img_tensor)
            gen_img = gen_img_tensor.cpu().numpy().squeeze().transpose((1, 2, 0))
            generated_images_B.append(gen_img)

        generated_images_A = []
        for img in real_images_B:
            img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).unsqueeze(0).float().to(model.device)
            gen_img_tensor = model.netG_B(img_tensor)
            gen_img = gen_img_tensor.cpu().numpy().squeeze().transpose((1, 2, 0))
            generated_images_A.append(gen_img)

    # Convert lists to numpy arrays
    generated_images_B = np.array(generated_images_B)
    generated_images_A = np.array(generated_images_A)

    # Calculate metrics
    avg_ssim_A2B, fid_A2B, inception_A2B = calculate_metrics(real_images_A, generated_images_B)
    avg_ssim_B2A, fid_B2A, inception_B2A = calculate_metrics(real_images_B, generated_images_A)

    print(f"A2B - SSIM: {avg_ssim_A2B}, FID: {fid_A2B}, Inception Score: {inception_A2B}")
    print(f"B2A - SSIM: {avg_ssim_B2A}, FID: {fid_B2A}, Inception Score: {inception_B2A}")

if __name__ == "__main__":
    main()
