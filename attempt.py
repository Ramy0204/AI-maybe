import glob, os
from PIL import Image
import numpy as np
import skeleton
def image(a, imgname):
    l = 28
    im = Image.new('RGB', (l,l), color = 0)
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            c = int(255*a[i + l*j])
            im.putpixel((i,j), (c,c,c))
    im.save(os.path.join('./data', f'{imgname}.png'))


def out(net: skeleton.Network, x):
    y = net.forward(x)
    return y.argmax(axis=0)
    
def guess(net: skeleton.Network, x):
    num = out(net, x)
    image(x, num)
    return num


def redraw(net :skeleton.Network, x):
    image(x, 'input')
    new = net.forward(x)
    image(new, 'output')
