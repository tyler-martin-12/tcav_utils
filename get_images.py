from bs4 import BeautifulSoup
import numpy as np
import requests
import cv2
import PIL.Image
import urllib

import matplotlib.pyplot as plt
import numpy as np
import sys

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

import os
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('-wnid', required=True, type=str, default=None)
parser.add_argument('-img_dir', required=True, type=str, default=None)
parser.add_argument('-max_images', required=False, type=int, default=1e7)
parser.add_argument('-start_idx', required=False, type=int, default=0)


args = parser.parse_args()

wnid = args.wnid
img_dir = os.path.abspath(args.img_dir)
max_images = args.max_images
start_idx = args.start_idx

# python get_images.py -wnid=n02391049 -img_dir='zebra' -max_images=10

def url_to_image(url):

    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
    return image


url = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=' + wnid
page = requests.get(url)

soup = BeautifulSoup(page.content, 'html.parser')

str_soup = str(soup)
split_urls=str_soup.split('\r\n')
total_images = len(split_urls)

img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)

n_images = max_images if total_images > max_images else total_images


did_write = 0

if not os.path.exists(img_dir):
	os.mkdir(img_dir)

for progress in range(n_images):

	if (progress%20==0):
		print(progress)

	save_path = os.path.join(img_dir,'img' + str(progress) + '.jpg')

	if not split_urls[progress] == None:
		if not os.path.exists(save_path):
			try:
				I = url_to_image(split_urls[progress])
				if len(I.shape) == 3:
					#print(save_path)
					cv2.imwrite(save_path,I)
					did_write += 1
			except:
				pass
		else:
			print('path already exists!')
			pass

print('wrote '+ str(did_write) + ' of ' + str(n_images) +' images')

