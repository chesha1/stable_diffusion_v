from PIL import Image, ImageDraw, ImageFont
num_of_imgs = 5
img1 = Image.open("image_0.png")
img2 = Image.open("image_1.png")
img3 = Image.open("image_2.png")
img4 = Image.open("image_3.png")
img5 = Image.open("image_4.png")
size = img1.size

img_text = Image.new("RGB", size=size, color=(255,255,255))
text = "lr=1e-3_step=2e3"
font = ImageFont.truetype("consola.ttf", 50, encoding="unic")
dr = ImageDraw.Draw(img_text)
dr.text((size[0]/8, size[1]/2), text=text, font=font, fill="#000000")


img_final = Image.new("RGB", ((num_of_imgs+1)*size[0], size[1]))
loc1 = (0, 0)
loc2 = (size[0], 0)
loc3 = (2*size[0], 0)
loc4 = (3*size[0], 0)
loc5 = (4*size[0], 0)
loc6 = (5*size[0], 0)

img_final.paste(img_text, loc1)
img_final.paste(img1, loc2)
img_final.paste(img2, loc3)
img_final.paste(img3, loc4)
img_final.paste(img4, loc5)
img_final.paste(img5, loc6)
img_final.save(text+".png")