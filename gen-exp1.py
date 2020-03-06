#coding: utf-8

import gplearn.GP as GP
from PIL import Image 

unidivfact = GP.GP_SymReg(500,100,0.01)
unidivfact.load("uni-divfact-D.model")
sizeworld = (1024,1024)

image = Image.new("RGB", sizeworld)
image.putpixel((x,y),(color))
image.save(str(i)+".png", "PNG")


z = map(float, z)
int(unidivfact.predict(z))


for x1 in range(1,sizeworld[0]):