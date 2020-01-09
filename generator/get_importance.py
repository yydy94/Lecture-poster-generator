from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import matplotlib.image as mpimg

path = '/home/sjtubcmi/桌面/data_text_region/20.jpg'

def draw_rectangle(draw, x1, y1, dx, dy, flag):
	width = 5
	color = (0, 255, 0)
	if flag:
		color = (255, 0, 0)
	draw.line((x1, y1, x1 + dx, y1), color, width)
	draw.line((x1, y1, x1, y1 + dy), color, width)
	draw.line((x1 + dx, y1, x1 + dx, y1 + dy), color, width)
	draw.line((x1, y1 + dy, x1 + dx, y1 + dy), color, width)


image = Image.open(path)
draw = ImageDraw.Draw(image)

draw_rectangle(draw, 379, 215, 245, 231, 1)
draw_rectangle(draw, 29, 634, 240, 452, 0)

image.show()
image.save('generate_data_harmony.jpg', quality = 75)
