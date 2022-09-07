import cv2.cv2 as cv2
import random
import numpy as np
from scipy import ndimage
from PIL import Image
from torchvision.transforms import transforms as T
import os
from itertools import cycle

class GraffitiApplier:
    def put_graffiti(self, sign_image, graffiti, min_gr_size, max_gr_size, min_alpha, max_alpha, left,
                     top, right, bottom, color):
        height, width, _ = sign_image.shape
        rotated = ndimage.rotate(graffiti, random.randint(0, 355), reshape=True, cval=255)

        img = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        tr = T.RandomPerspective(distortion_scale=0.5, p=1.0, fill=255)
        img_pil_perspective = tr(im_pil)
        rotated = np.asarray(img_pil_perspective)
        gr_size = (random.randint(min_gr_size, max_gr_size), random.randint(min_gr_size, max_gr_size))
        graffiti = cv2.resize(rotated, gr_size)

        y_offset = random.randint(top, bottom - gr_size[1])
        x_offset = random.randint(left, right - gr_size[0])

        roi = sign_image[y_offset:gr_size[1] + y_offset, x_offset:gr_size[0] + x_offset]
        small_img_gray = cv2.cvtColor(graffiti, cv2.COLOR_RGB2GRAY)
        ret, mask = cv2.threshold(small_img_gray, 200, 255, cv2.THRESH_BINARY)

        mask = mask <= 30
        gr_roi = roi.copy()
        gr_roi[mask] = np.array(color)
        alpha = random.uniform(min_alpha, max_alpha)
        overlay = roi.copy()
        output = gr_roi.copy()
        cv2.addWeighted(overlay, alpha, output, 1 - alpha,
                        0, output)

        sign_image[y_offset: y_offset + output.shape[0], x_offset: x_offset + output.shape[1]] = output
        return sign_image

    def put_nakleiki(self, sign_image, graffiti, min_gr_size, max_gr_size, left, top, right, bottom):
        height, width, _ = sign_image.shape
        graffiti.fill(255)
        rotated = ndimage.rotate(graffiti, random.randint(0, 355), reshape=True, cval=0)
        img = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        tr = T.RandomPerspective(distortion_scale=0.3, p=1.0)
        img_pil_perspective = tr(im_pil)
        rotated = np.asarray(img_pil_perspective)
        gr_size = (random.randint(min_gr_size, max_gr_size), random.randint(min_gr_size, max_gr_size))
        graffiti = cv2.resize(rotated, gr_size)

        y_offset = random.randint(top, bottom-gr_size[1])
        x_offset = random.randint(left, right-gr_size[0])

        roi = sign_image[y_offset:gr_size[1] + y_offset, x_offset:gr_size[0] + x_offset]
        small_img_gray = cv2.cvtColor(graffiti, cv2.COLOR_RGB2GRAY)
        mask = small_img_gray > 0

        color = random.choice([[10,10,10],[245,245,245],[0,150,0],[205,0,0]])
        gr_roi = roi.copy()
        gr_roi[mask] = np.array(color)

        sign_image[y_offset: y_offset + gr_roi.shape[0], x_offset: x_offset + gr_roi.shape[1]] = gr_roi
        return sign_image

class GraffitiProcessor(GraffitiApplier):
    def __init__(self, path2data, pth2gr, colors, stick_stickers: bool = True):
        self.path_2_gosts = [gost for gost in os.listdir(path2data)]
        self.bw_imgs = [os.path.join(pth2gr, i) for i in os.listdir(os.path.join(pth2gr))]
        self.colors_cycle = cycle(colors)
        self.color = next(self.colors_cycle)
        self.stick_stickers = stick_stickers

    def choose_rand_gr(self):
        gr_type = random.choice(['graffiti', 'nakleiki'])
        rand_gr_path = random.choice(self.bw_imgs)
        if self.stick_stickers:
            return rand_gr_path, gr_type
        else:
            return rand_gr_path, 'graffiti'

    def apply_single_transform(self, gr_type, kwargs):
        if gr_type == "graffiti":
            kwargs['min_alpha'], kwargs['max_alpha'] = 0.25, 0.6
            kwargs['color'] = self.color

            new_image = self.put_graffiti(**kwargs)
            return new_image

        elif gr_type == "nakleiki":
            new_image = self.put_nakleiki(**kwargs)
            return new_image

