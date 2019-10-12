import os
import sys
import uuid
import math
from PIL import Image

def resize_preserve_aspect(im, target_width, target_height):
    original_width = im.size[0]
    original_height = im.size[1]
    original_ratio = original_width / original_height
    target_ratio = target_width / target_height
    if target_ratio > original_ratio:
        new_size = (target_width, math.ceil(target_width / original_ratio))
    else:
        new_size = (math.ceil(target_height * original_ratio), target_height)
    # if the ratio is only slightly off, just snap to the target dimensions
    if (new_size[0] > target_width and new_size[0] < target_width + 3) or (new_size[1] > target_height and new_size[1] < target_height + 3):
        new_size = (target_width, target_height) 
    return im.resize(new_size, Image.ANTIALIAS)

def fix_aspect_ratio(im, target_width, target_height):
    if im.size[0] == target_width and im.size[1] == target_height:
        return im
    elif im.size[0] > target_width:
        offset = int(math.floor((im.size[0] - target_width) / 2))
        return im.crop((offset, 0, im.size[0] - offset, target_height))
    elif im.size[1] > target_height:
        offset = int(math.floor((im.size[1] - target_height) / 2))
        return im.crop((0, offset, target_width, im.size[1] - offset))
    raise Exception("unreachable code")


class ImageNormalizer(object):
    def __init__(self, target_width, target_height):
        self.target_height = target_height
        self.target_width = target_width

    def normalize_image(self, im):
        normalized_images = []
        patches = []

        # scale down the image
        resized = self._resize(im)
        patches.append(self._fix_aspect_ratio(resized))

        # extract patches from full sized image
        scale_factor_double_width = min(im.size[0] // (2*self.target_width), im.size[1] // (2*self.target_height))
        patches += self._extract_patches(im, scale_factor=scale_factor_double_width)
        scale_factor_triple_width = min(im.size[0] // (3*self.target_width), im.size[1] // (3*self.target_height))
        if not (scale_factor_double_width == scale_factor_triple_width):
            patches += self._extract_patches(im, scale_factor=scale_factor_triple_width)

        mirrors = self._mirror(patches)
        for cropped_image in mirrors:
            if cropped_image.size[0] < self.target_width or cropped_image.size[1] < self.target_height:
                raise Exception("Bad size")
            if cropped_image.size[0] != self.target_width or cropped_image.size[1] != self.target_height:
                cropped_image = cropped_image.crop((0, 0, self.target_width, self.target_height))
            desaturated_crop = self._desaturate(cropped_image)
            normalized_images.append({"colour": self._standardize_rgb(cropped_image), "mono": desaturated_crop})
        return normalized_images
                
    def _mirror(self, ims):
        output = []
        for im in ims:
            output.append(im)
            output.append(im.transpose(Image.FLIP_LEFT_RIGHT))
        return output

    def _resize(self, im):
        return resize_preserve_aspect(im, self.target_width, self.target_height) 
    
    def _desaturate(self, im):
        matrix = (
            0.5, 0.5, 0, 0,
            0.5, 0.5, 0, 0,
            0.5, 0.5, 0, 0)
        if im.mode == "RGB":
            return im.convert("RGB", matrix).convert("L")
        elif im.mode == "RGBA":
            rgb_im = im.convert("RGB")
            return rgb_im.convert("RGB", matrix).convert("L")
        else:
            return im.convert("L")

    def _standardize_rgb(self, im):
        if im.mode != "RGB":
            return im.convert("RGB")
        return im

    def _fix_aspect_ratio(self, im):
        return fix_aspect_ratio(im, self.target_width, self.target_height)

    def _extract_patches(self, large_im, scale_factor=None):
        im = large_im
        if scale_factor:
            im = large_im.resize((large_im.size[0] // scale_factor, large_im.size[1] // scale_factor))
        if im.size[0] < self.target_width * 2 and im.size[1] < self.target_height * 2:
            return []
        patches = []
        for x in range(0, im.size[0] - self.target_width + 1, self.target_width // 2):
            for y in range(0, im.size[1] - self.target_height + 1, self.target_height // 2):
                patches.append(im.crop((x, y, x + self.target_width, y + self.target_width)))
        return patches