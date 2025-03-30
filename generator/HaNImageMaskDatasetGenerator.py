
# Copyright 2024 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# HaNImageMaskDatasetGenerator.py
# 2024/05/14

import os
import sys
import cv2
import nrrd
import numpy as np

import shutil
import glob
import traceback

from ConfigParser import ConfigParser
from PIL import Image, ImageOps

class HaNImageMaskDatasetGenerator:

  def __init__(self, config_file):
    print("=== HaNImageMaskDatasetGenerator")
    self.config      = ConfigParser(config_file)
    self.config.dump_all()
    self.images_dir   = self.config.get(ConfigParser.GENERATOR, "images_dir", 
                                        dvalue ="./PDDC/*/*/")
    self.image_file   = self.config.get(ConfigParser.GENERATOR, "image_file",  
                                        dvalue= "img.nrrd")
    self.masks_dir    = self.config.get(ConfigParser.GENERATOR, "masks_dir", 
                                        dvalue ="./PDDC/*/*/structure/")
    self.category    = self.config.get(ConfigParser.GENERATOR, "category", dvalue="Mandible")
    self.mask_file   = self.category + ".nrrd"
    output_dir       = self.config.get(ConfigParser.GENERATOR, "output_dir")
    self.output_dir  = output_dir.format(self.category)
    print("--- self.output_dir {}".format(self.output_dir))
    self.index_order = 'F'
    self.resize      = self.config.get(ConfigParser.GENERATOR, "resize", dvalue=(512, 512))
    #     
    if os.path.exists(self.output_dir):
       shutil.rmtree(self.output_dir)
    if not os.path.exists(self.output_dir):
       os.makedirs(self.output_dir)

    self.output_images_dir = self.output_dir + "/images/"
    self.output_masks_dir  = self.output_dir + "/masks/"
    if not os.path.exists(self.output_images_dir):
       os.makedirs(self.output_images_dir)
    if not os.path.exists(self.output_masks_dir):
       os.makedirs(self.output_masks_dir)

    self.validation = self.config.get(ConfigParser.VALIDATOR, "validation")
    valid_dir  = self.config.get(ConfigParser.VALIDATOR, "output_dir")
    self.valid_dir = valid_dir.format(self.category)

    if os.path.exists(self.valid_dir):
       shutil.rmtree(self.valid_dir)
    if not os.path.exists(self.valid_dir):
       os.makedirs(self.valid_dir)

  def generate(self): 
    print("--- self.images_dir {}".format(self.images_dir))
    subdirs = os.listdir(self.images_dir)
    print("=== subdirs {}".format(subdirs))
    self.subdir_index = 1000
    for subdir in subdirs:
       self.subdir_index += 1
       print("--- Subdir {}".format(subdir))
       subdir_path = os.path.join(self.images_dir, subdir)
       if os.path.isdir(subdir_path):
         subsub_dirs = os.listdir(subdir_path)
         subsub_dirs = sorted(subsub_dirs)
         for subsub_dir in subsub_dirs:
           subsub_dirpath = os.path.join(subdir_path, subsub_dir)
           self.generate_one(subsub_dirpath )
       else:
         print("--- Skipping a file  {}".format(subdir_path))

  def generate_one(self, subdir_path):
    print("=== generate_one {}".format(subdir_path))
    mask_file = subdir_path + "/structures/" + self.category + ".nrrd"
    print("--- mask_file {}".format(mask_file))

    seg_files = glob.glob(mask_file)
    seg_files = sorted(seg_files)
    print("--- generate_one subidr:{} len: {}".format(subdir_path, len(seg_files)))
    
    file_index = 1000
    for seg_file in seg_files:
      file_index += 1
      self.read_nrrd_mask_file(seg_file, file_index)

    ct_image_files = glob.glob(subdir_path + "/" + self.image_file)
    file_index = 1000
    for ct_image_file in ct_image_files:
      file_index += 1
      self.read_nrrd_image_file(ct_image_file, file_index)

  def read_nrrd_mask_file(self, filename, file_index):
    data, header = nrrd.read(filename, index_order=self.index_order)
    #print(data.shape)
    len = data.shape[2]
    for i in range(len):
      image = data[:,:,i]
      if image.any() > 0:
        output_filename= str(self.subdir_index) + "_" + str(file_index) + "_" + str(i+100) + ".jpg"
        output_filepath = os.path.join(self.output_masks_dir, output_filename)
        #image = cv2.resize(image, self.resize)
        image = image * 255
        cv2.imwrite(output_filepath, image)
        print("--- Saved {}".format(output_filepath))

  def read_nrrd_image_file(self, filename, file_index):
    data, header = nrrd.read(filename, index_order=self.index_order)
    len = data.shape[2]
    for i in range(len):
      image = data[:,:,i]
      output_filename= str(self.subdir_index) + "_" + str(file_index) + "_" + str(i+100) + ".jpg"
      mask_filepath  = os.path.join(self.output_masks_dir, output_filename)
      if os.path.exists(mask_filepath):
        output_filepath = os.path.join(self.output_images_dir, output_filename)
        cv2.imwrite(output_filepath, image)
        print("--- Saved {}".format(output_filepath))

  def pil2cv(self, image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2: 
        pass
    elif new_image.shape[2] == 3: 
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4: 
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

  def validate(self):
    if self.validation==False:
      return
    image_files = glob.glob(self.output_images_dir + "/*.jpg")
    mask_files  = glob.glob(self.output_masks_dir  + "/*.jpg")
    image_files = sorted(image_files)
    mask_files  = sorted(mask_files)
    num_images = len(image_files)
    num_masks  = len(mask_files)

    if num_images != num_masks:
      raise Exception("Error ")
    for i in range(num_images):
      image_file = image_files[i]
      mask_file  = mask_files[i]

      image = Image.open(image_file)
      image = image.convert("RGB")
      image = self.pil2cv(image)
      #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      mask  = Image.open(mask_file)
      mask  = mask.convert("L")
      #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
      mask =  ImageOps.colorize(mask, black="black", white="green")
      mask = mask.convert("RGB")
      mask = self.pil2cv(mask)
      image += mask
      blended_filepath = os.path.join(self.valid_dir, str(i+1000) + ".jpg")
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      cv2.imwrite(blended_filepath, image)
      print("--- saved {}".format(blended_filepath))

if __name__ == "__main__":
  try:
    config_file = "./generator.config"
    if len(sys.argv) ==2:
      config_file = sys.argv[1]

    generator = HaNImageMaskDatasetGenerator(config_file)
    generator.generate()
    generator.validate()

  except:
    traceback.print_exc()

