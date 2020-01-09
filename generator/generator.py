from PIL import Image, ImageDraw, ImageFont
from cv.first.eval import eval_importance
from cv.second.eval import eval_harmony
import glob
import numpy as np
import torch
import matplotlib.image as mpimg

path = '/home/sjtubcmi/桌面/data_text_region/2.jpg'
fontPath = '/usr/share/fonts/truetype/lao/Phetsarath_OT.ttf'
image = Image.open(path)
vis = np.zeros(image.size)

def calcRectangle(x1, x2, y, image, text, size, max_line):
	draw = ImageDraw.Draw(image)
	setFont = ImageFont.truetype(fontPath, size)
	nowx = x1
	nowy = y
	nowmax = 0
	ch = 0
	lines = 1
	while ch < len(text):
		ch_r = ch + 1
		if text[ch] != ' ' and text[ch] != '\n':
			while(ch_r < len(text) and text[ch_r] != ' ' and text[ch_r] != '\n'):
				ch_r = ch_r + 1
		width, height = draw.textsize(text[ch: ch_r], setFont)
		if x1 + width > x2 :
			return -1
		if nowx + width > x2:
			nowx = x1
			nowy = nowy + nowmax
			nowmax = 0
			lines = lines + 1
			if lines > max_line:
				return -1
			if text[ch] == ' ' or text[ch] == '\n':
				ch = ch + 1
			continue
		nowmax = max(nowmax, height)
		nowx = nowx + width
		ch = ch_r
	return nowy + nowmax
	
def draw_rectangle(draw, x1, x2, y1, y2):
	draw.line((x1, y1, x2, y1), (0, 255, 0))
	draw.line((x1, y1, x1, y2), (0, 255, 0))
	draw.line((x2, y1, x2, y2), (0, 255, 0))
	draw.line((x1, y2, x2, y2), (0, 255, 0))

def addAtPos(x1, x2, y, image, text, size, color):
	draw = ImageDraw.Draw(image)
	setFont = ImageFont.truetype(fontPath, size)
	nowx = x1
	nowy = y
	nowmax = 0
	ch = 0
	while ch < len(text):
		ch_r = ch + 1
		if text[ch] != ' ' and text[ch] != '\n':
			while(ch_r < len(text) and text[ch_r] != ' ' and text[ch_r] != '\n'):
				ch_r = ch_r + 1
		width, height = draw.textsize(text[ch: ch_r], setFont)
		if x1 + width > x2 :
			return -1
		if nowx + width > x2:
			draw.text((nowx, nowy), '\n' , font = setFont, fill = color)
			nowx = x1
			nowy = nowy + nowmax
			nowmax = 0
			if text[ch] == ' ' or text[ch] == '\n':
				ch = ch + 1
			continue
		nowmax = max(nowmax, height)
		draw.text((nowx, nowy), text[ch: ch_r], font = setFont, fill = color)
		nowx = nowx + width
		ch = ch_r
	
	y1 = y
	y2 = nowy + nowmax
	
	#draw_rectangle(draw, x1, x2, y1, y2)

x_1 = []
x_2 = []
y_1 = []
y_2 = []

def is_rect_intersect(x01, x02, y01, y02, x11, x12, y11, y12):
	zx = abs(x01 + x02 -x11 - x12)
	x  = abs(x01 - x02) + abs(x11 - x12)
	zy = abs(y01 + y02 - y11 - y12)
	y  = abs(y01 - y02) + abs(y11 - y12)
	if zx <= x and zy <= y:
		return 1
	return 0

def check(x1, x2, y1, y2):
	for i in range(len(x_1)):
		if is_rect_intersect(x1, x2, y1, y2, x_1[i], x_2[i], y_1[i], y_2[i]):
			return 1
	return 0

def setVis(x1, x2, y1, y2):
	x_1.append(x1)
	x_2.append(x2)
	y_1.append(y1)
	y_2.append(y2)

lower_bound = [0.5, 1.5, 2.5, 3.5]

def addText(image, text, size, importance, max_line = 10000, color = (0, 0, 0), other = 0):
	rgb = torch.Tensor(mpimg.imread(path)).float()
	width, height = image.size
	st_dx = 1
	dx_list = []
	for dx in range(1, width, 10):
		dx_list.append(calcRectangle(0, dx, 0, image, text, size, max_line))
	for x in range(width // 10, width - width // 10, 10):
		for dx in range(st_dx, width - x, 10):
			dy = dx_list[dx // 10]
			if dy == -1:
				continue
			for y in range(height // 10, min(height - dy, height - height // 10), 10):
				x1 = x
				x2 = x + dx
				y1 = y
				y2 = y1 + dy
				
				if check(x1, x2, y1, y2):
					continue
				
				this_import = eval_importance(x1, y1, dx, y2 - y1, width, height)
				this_harm = eval_harmony(x1, y1, dx, y2 - y1, rgb)
				if this_import * 4 < lower_bound[importance - 1]:
					continue
				if this_harm * 2 >= 0.6:
					continue
				
				addAtPos(x1, x2, y1, image, text, size, color)
				setVis(x1, x2 + other, y1, y2 + other)
				
				draw = ImageDraw.Draw(image)
				
				print(x1, x2, y1, y2)
				
				print(this_import)
				
				print(this_harm)
				
				return

def work(path):
	addText(image, '2019.03.20', 35, 3, 1)
	addText(image, 'Marcus Textor', 30, 3, 1)
	addText(image, 'Bioinspired Surfaces for Application in the Life Sciences: Innovation, Technology and Translation to Application', 40, 4, 4, color = (255, 155, 0), other = 0)
	addText(image, 'All biomaterials placed into physiological systems are treated as foreign bodies: in soft or hard tissue, or within the blood stream. Host responses to materials in the body (“in vivo”) are diverse, complex and unavoidable. Many occur at the interface between living biology and material.', 20, 1)
	addText(image, 'Regulatory affairs are crucial (success or failure, very different if considered as a medical de-vice or a drug (e.g. devices with drug delivery function). The talk will cover three exemplary cases: (a) Dental root titanium implant surfaces, (b) Trig-gered liposomal drug release system, (c) Fetal Surgery and Fetal Membrane Sealing.', 20, 1)	
	
	image.show()
	
	image.save('generate4.jpg', quality = 95)
	

work(path)
