from PIL import Image, ImageDraw, ImageFont
from cv.first.eval import eval_importance
from cv.second.eval import eval_harmony
import glob
import numpy as np
import torch
import random
import matplotlib.image as mpimg

path = '/home/sjtubcmi/桌面/data_text_region/3.jpg'
fontPath = '/usr/share/fonts/truetype/lao/Phetsarath_OT.ttf'
image = Image.open(path)

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
	
def draw_rectangle(draw, x1, x2, y1, y2, color = (0, 255, 0)):
	print(color)
	width = 3
	draw.line((x1, y1, x2, y1), color, width)
	draw.line((x1, y1, x1, y2), color, width)
	draw.line((x2, y1, x2, y2), color, width)
	draw.line((x1, y2, x2, y2), color, width)

cnt = 0

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
	global cnt
	#draw_rectangle(draw, x1, x2, y1, y2, ((cnt & 1) * 255, ((cnt >> 1) & 1) * 255, ((cnt >> 2) & 1) * 255))

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

lower_bound = [0, 1.5, 2, 3]
upper_bound = [2, 2.5, 3.5, 4]

test = 0
alpha = 10
beta = 1
gamma = 1000
def calcVal(y1, v1, v2, v3):
	return alpha * v1 - beta * v2 - gamma * v3 - test * y1

def addText(image, text, size, importance, max_line = 10000, color = (0, 0, 0), other = 0, aligned_x = -1, aligned_y = -1):
	global cnt
	cnt = cnt + 1
	print([cnt])
	print([cnt & 1, (cnt >> 1) & 1, (cnt >> 2) & 1])
	rgb = torch.Tensor(mpimg.imread(path)).float()
	width, height = image.size
	print(max_line)
	st_dx = max(0, 2 - importance) * 200
	Rec = []
	for dx in range(st_dx, width - width // 10, 20):
		dy = calcRectangle(0, dx, 0, image, text, size, max_line)
		if dy == -1:
			continue
		for x in range(width // 10, width - width // 10 - dx, 20):
			for y in range(height // 20, min(height - dy, height - height // 20), 20):
				x1 = x
				x2 = x + dx
				y1 = y
				y2 = y1 + dy
				
				this_import = eval_importance(x1, y1, dx, y2 - y1, width, height)
				this_harm = eval_harmony(x1, y1, dx, y2 - y1, rgb)
				if this_import * 4 < lower_bound[importance - 1]:
					continue
				if this_import * 4 > upper_bound[importance - 1]:
					continue
				if this_harm * 2 >= 0.6:
					continue
				
				draw = ImageDraw.Draw(image)
				
				Rec.append((x1, y1, x2, y2, this_import, this_harm))
	
	if len(Rec) == 0:		
		print('impossible')
		return Rec
	
	#for i in range(2):
	#	x = random.randint(0, len(Rec) - 1)
	#	draw_rectangle(draw, Rec[x][0], Rec[x][2], Rec[x][1], Rec[x][3], ((cnt & 1) * 255, ((cnt >> 1) & 1) * 255, ((cnt >> 2) & 1) * 255))
	#	print([Rec[x][4], Rec[x][5]])
	#return (1, 2)
	
	print(len(Rec))
	
	nowpos = -1
	nowdelta = 0
	for i in range(1, len(Rec)):
		x1 = Rec[i][0]
		y1 = Rec[i][1]
		x2 = Rec[i][2]
		y2 = Rec[i][3]
		
		if check(x1, x2, y1, y2):
			continue
		
		if nowpos == -1:
			nowpos = i
			if aligned_x != -1:
				nowdelta = abs(aligned_x - x1) // 5
			
			if aligned_y != -1:
				nowdelta = abs(aligned_y - y1) // 5
			continue
		
		delta = 0
		
		if aligned_x != -1:
			delta = abs(aligned_x - x1) // 5
		
		if aligned_y != -1:
			delta = abs(aligned_y - y1) // 5
		
		if calcVal(Rec[i][1] // 50 * max(importance - 0.5, 0), Rec[i][4], Rec[i][5], delta) > calcVal(Rec[nowpos][1] // 50 * max(importance - 0.5, 0), Rec[nowpos][4], Rec[nowpos][5], nowdelta):
			nowpos = i
			nowdelta = delta
	
	x1 = Rec[nowpos][0]
	y1 = Rec[nowpos][1]
	x2 = Rec[nowpos][2]
	y2 = Rec[nowpos][3]
	
	if check(x1, x2, y1, y2):
		print('impossible')
		return Rec
	
	addAtPos(x1, x2, y1, image, text, size, color)
	
	setVis(x1, x2, y1, y2 + max(importance - 3, 0) * 50)
	
	print(Rec[nowpos][4], Rec[nowpos][5])
	
	print(x1, y1, x2, y2)
	
	return (x1, y1, x2, y2)

def work(path):
	Rec1 = addText(image, '2019.03.20', 35, 3, 1, aligned_y = 0)
	print(len(Rec1))
	Rec2 = addText(image, 'Marcus Textor', 35, 3, 1, aligned_y = 0)
	print(len(Rec2))
	Rec3 = addText(image, 'Bioinspired Surfaces for Application in the Life Sciences: Innovation, Technology and Translation to Application', 40, 4, 5, color = (255, 155, 0), other = 0)
	print(len(Rec3))
	Rec4 = addText(image, 'All biomaterials placed into physiological systems are treated as foreign bodies: in soft or hard tissue, or within the blood stream. Host responses to materials in the body (“in vivo”) are diverse, complex and unavoidable. Many occur at the interface between living biology and material.', 20, 1)
	print(len(Rec4))
	Rec5 = addText(image, 'Regulatory affairs are crucial (success or failure, very different if considered as a medical de-vice or a drug (e.g. devices with drug delivery function). The talk will cover three exemplary cases: (a) Dental root titanium implant surfaces, (b) Trig-gered liposomal drug release system, (c) Fetal Surgery and Fetal Membrane Sealing.', 20, 1)	
	print(len(Rec5))
	
	image.show()
	
	image.save('tmp1.jpg', quality = 95)
	

work(path)
