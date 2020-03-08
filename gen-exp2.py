

import gplearn.GP as GP
from PIL import Image 
from multiprocessing import Pool

unidivfact = GP.GP_SymReg(500,100,0.01)
unidivfact.load("uni-divfact-D.model")
posflatconv = GP.GP_SymReg(500,100,0.01)
posflatconv.load("pos-flat-convertor.model")


sizeworld = (1024,256)
szd = int((sizeworld[0] * sizeworld[1])/2)


image = Image.new("RGB", sizeworld)

def func1(i):
	for x in range(1,sizeworld[0]):
		for y in range(1,sizeworld[1]):
			print((i/255.0)*100.0,"% - ",(x/sizeworld[0])*100.0,"% - ",(y/sizeworld[1])*100.0,"%")
			z2 = [x,y,i]
			z2 = list(map(float, z2))
			y2 = int(posflatconv.predict(z2))
			
			z1 = [i%16,y2]
			z1 = list(map(float, z1))
			y1 = int(unidivfact.predict(z1))

			if y1 != 0:
				image.putpixel((x,y),(((i%8)*y2)%256,((i%16)*y2)%256,((i%4)*y2)%256))
			
with Pool(16) as p:
	print(p.map(func1, range(1,255)))			

image.save("g-exp3-0.png", "PNG")