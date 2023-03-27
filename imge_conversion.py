from PIL import Image
from util import filesNdirec
root_dir = 'dataset/CLUST2D/train/'
a,b,c = filesNdirec(root_dir,'.png')

for img in a:

    im = Image.open(img)
    new_name =  str(img).replace('.png','.jpg')
    print(new_name)
    im.convert('RGB').save(new_name)


