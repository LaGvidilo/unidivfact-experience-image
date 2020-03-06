#coding: utf-8

import gplearn.GP as GP
from PIL import Image 

unidivfact = GP.GP_SymReg(500,100,0.01)
unidivfact.load("uni-divfact-D.model")
posflatconv = GP.GP_SymReg(500,100,0.01)
posflatconv.load("pos-flat-convertor.model")


sizeworld = (512,512)
szd = int((sizeworld[0] * sizeworld[1])/2)


image = Image.new("RGB", sizeworld)

for i in range(1,255):
	for x in range(1,sizeworld[0]):
		for y in range(1,sizeworld[1]):
			print((i/255.0)*100.0,"% - ",(x/sizeworld[0])*100.0,"% - ",(y/sizeworld[1])*100.0,"%")
			z2 = [x,y,szd]
			z2 = list(map(float, z2))
			y2 = int(posflatconv.predict(z2))

			z1 = [i,y2]
			z1 = list(map(float, z1))
			y1 = int(unidivfact.predict(z1))
			if y1 > 0:image.putpixel((x,y),(y1*i,y1*i,y1*i))

image.save("g-exp1.png", "PNG")