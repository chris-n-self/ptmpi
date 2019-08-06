# 12/01/18
# Chris Self

# python3 compatibility (http://python-future.org/imports.html)
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np

#
# CONVERT COLOUR SPACES
# use rgb as the default everywhere, since that is what matplotlib handles
#

def norm_rgb(*args):
	if isinstance(args[0],tuple):
		# assume argument was passed as single tuple obj
		r,g,b = args[0]
	else:
		r,g,b = args
	return r/255.,g/255.,b/255.

def int_rgb(*args):
	if isinstance(args[0],tuple):
		# assume argument was passed as single tuple obj
		r,g,b = args[0]
	else:
		r,g,b = args
	return np.ceil(r*255.),np.ceil(g*255.),np.ceil(b*255.)

def cmyk_to_rgb(*args):
	if isinstance(args[0],tuple):
		c,m,y,k = args[0]
	else:
		c,m,y,k = args

	r = 255.*(1-c)*(1-k)
	g = 255.*(1-m)*(1-k)
	b = 255.*(1-y)*(1-k)
	return norm_rgb(r,g,b)

def rgb_to_cmyk(*args):
	if isinstance(args[0],tuple):
		r,g,b = args[0]
	else:
		r,g,b = args

	k = 1.-max(r,g,b)
	c = (1.-r-k) / (1.-k)
	m = (1.-g-k) / (1.-k)
	y = (1.-b-k) / (1.-k)
	return c,m,y,k

"""
convert rgb to ryb
adapted from http://www.deathbysoftware.com/colors/index.html
"""
def rgb_to_ryb(*args):
	if isinstance(args[0],tuple):
		# assume argument was passed as single tuple obj
		iRed,iGreen,iBlue = int_rgb(args[0])
	else:
		iRed,iGreen,iBlue = int_rgb(args)

	# Remove the white from the color
	iWhite = min(iRed, iGreen, iBlue)	
	iRed = iRed - iWhite
	iGreen = iGreen - iWhite
	iBlue = iBlue - iWhite
		
	iMaxGreen = max(iRed, iGreen, iBlue)
		
	# Get the yellow out of the red+green		
	iYellow = min(iRed, iGreen)
	iRed = iRed - iYellow
	iGreen = iGreen - iYellow
		
	# If this unfortunate conversion combines blue and green, then cut each in half to
	# preserve the value's maximum range.
	if (iBlue > 0 and iGreen > 0):
		iBlue = iBlue/2.
		iGreen = iGreen/2.
		
	# Redistribute the remaining green.
	iYellow = iYellow + iGreen
	iBlue = iBlue + iGreen
		
	# Normalize the values.
	iMaxYellow = max(iRed, iYellow, iBlue)
	if (iMaxYellow > 0):
		iN = iMaxGreen / iMaxYellow
		iRed = iRed * iN
		iYellow = iYellow * iN
		iBlue = iBlue * iN
		
	# Add the white back in.
	iRed = iRed + iWhite
	iYellow = iYellow + iWhite
	iBlue = iBlue + iWhite
		
	r,y,b = np.floor(iRed),np.floor(iYellow),np.floor(iBlue)
	return norm_rgb(r,y,b)

"""
convert ryb to rgb
adapted from https://github.com/bahamas10/ryb
"""
MAGIC_COLORS = np.array([[1.,1.,1.],\
						 [1.,1.,0.],\
						 [1.,0.,0.],\
						 [1.,0.5,0.],\
						 [0.163,0.373,0.6],\
						 [0.,0.66,0.2],\
						 [0.5,0.,0.5],\
						 [0.2,0.094,0.]])

def cubicInt(t, A, B):
	weight = t * t * (3. - 2. * t)
	return A + weight * (B - A)

def getR(iR, iY, iB):
	magic = MAGIC_COLORS
	# red
	x0 = cubicInt(iB, magic[0][0], magic[4][0])
	x1 = cubicInt(iB, magic[1][0], magic[5][0])
	x2 = cubicInt(iB, magic[2][0], magic[6][0])
	x3 = cubicInt(iB, magic[3][0], magic[7][0])
	y0 = cubicInt(iY, x0, x1)
	y1 = cubicInt(iY, x2, x3)
	return cubicInt(iR, y0, y1)

def getG(iR, iY, iB):
	magic = MAGIC_COLORS
	# green
	x0 = cubicInt(iB, magic[0][1], magic[4][1])
	x1 = cubicInt(iB, magic[1][1], magic[5][1])
	x2 = cubicInt(iB, magic[2][1], magic[6][1])
	x3 = cubicInt(iB, magic[3][1], magic[7][1])
	y0 = cubicInt(iY, x0, x1)
	y1 = cubicInt(iY, x2, x3)
	return cubicInt(iR, y0, y1)

def getB(iR, iY, iB):
	magic = MAGIC_COLORS
	# blue
	x0 = cubicInt(iB, magic[0][2], magic[4][2])
	x1 = cubicInt(iB, magic[1][2], magic[5][2])
	x2 = cubicInt(iB, magic[2][2], magic[6][2])
	x3 = cubicInt(iB, magic[3][2], magic[7][2])
	y0 = cubicInt(iY, x0, x1)
	y1 = cubicInt(iY, x2, x3)
	return cubicInt(iR, y0, y1)

def ryb_to_rgb(*args):
	if isinstance(args[0],tuple):
		# assume argument was passed as single tuple obj
		r,y,b = args[0]
	else:
		r,y,b = args

	R1 = getR(r, y, b)
	G1 = getG(r, y, b)
	B1 = getB(r, y, b)
	r,g,b = np.ceil(R1 * 255.),np.ceil(G1 * 255.),np.ceil(B1 * 255.)
	return norm_rgb(r,g,b)

def hex_to_rgb(hexval):
	hexval = hexval.lstrip('#')
	r,g,b = tuple(int(hexval[i:i+2], 16) for i in (0, 2 ,4))
	return norm_rgb(r,g,b)

#
# COMPLEMENTARY COLOURS
# taking the complementary colour is easy, it is just 1-.. for all of the tonal parts of the colour.
# however, the comp transform works differently in the different colour spaces. these functions just
# wrap the transforms back and forth to the right colour space to take the comp
# 

def rgb_comp(*args):
	if isinstance(args[0],tuple):
		# assume argument was passed as single tuple obj
		r,g,b = args[0]
	else:
		r,g,b = args
	return 1.-r,1.-g,1.-b

def cmyk_comp(*args):
	if isinstance(args[0],tuple):
		# assume argument was passed as single tuple obj
		c,m,y,k = rgb_to_cmyk(args[0])
	else:
		c,m,y,k = rgb_to_cmyk(args)
	return cmyk_to_rgb(1.-c,1.-m,1.-y,k)

def ryb_comp(*args):
	if isinstance(args[0],tuple):
		# assume argument was passed as single tuple obj
		r,y,b = rgb_to_ryb(args[0])
	else:
		r,y,b = rgb_to_ryb(args)
	return ryb_to_rgb(1.-r,1.-y,1.-b)

#
# LAYER RGB (e.g. LIGHTEN/DARKEN)
# following this stackexchange answer:
# https://stackoverflow.com/questions/6615002/given-an-rgb-value-how-do-i-create-a-tint-or-shade
#

def layer(c1,c2):
	"""
	inputs should be: (r,g,b),(ar,ag,ab,alpha)

	the return matches the format of the first (c1) argument
	i.e. if (r,g,b) are integers in the range [0,255] then the answer is also returned in that 
	format, else if (r,g,b) are real numbers in [0.,1.] the answer is returned in that format
	"""

	# want the (a)r,(a)g,(a)b values as integers in the range [0,255]
	currentR,currentG,currentB = c1
	_passed_normed = False
	if not type(currentR)==type(1):
		_passed_normed = True
		currentR,currentG,currentB = int_rgb(currentR,currentG,currentB)
	aR,aG,aB,alpha = c2
	if not type(aR)==type(1):
		aR,aG,aB = int_rgb(aR,aG,aB)

	# apply layering
	newR = currentR + (aR - currentR) * alpha
	newG = currentG + (aG - currentG) * alpha
	newB = currentB + (aB - currentB) * alpha
	
	# return answer in the format the function recieved c1
	if _passed_normed:
		newR,newG,newB = norm_rgb(newR,newG,newB)
	return newR,newG,newB

def darken(rgb,alpha):
	return layer(rgb,(0,0,0,alpha))

def lighten(rgb,alpha):
	return layer(rgb,(255,255,255,alpha))

#
# DEFINE COLOUR SCALES
#

# cmyk and ryb vars hold the primary, secondard and tertiary colours of each 

def cmyk_wheel(k):
	cmyk = [cmyk_to_rgb(1.,0.,0.,k),cmyk_to_rgb(1.,0.5,0.,k),cmyk_to_rgb(1.,1.,0.,k),cmyk_to_rgb(0.5,1.,0.,k),\
			cmyk_to_rgb(0.,1.,0.,k),cmyk_to_rgb(0.,1.,0.5,k),cmyk_to_rgb(0.,1.,1.,k),cmyk_to_rgb(0.,0.5,1.,k),\
			cmyk_to_rgb(0.,0.,1.,k),cmyk_to_rgb(0.5,0.,1.,k),cmyk_to_rgb(1.,0.,1.,k),cmyk_to_rgb(1.,0.,0.5,k)]
	return cmyk
cmyk = cmyk_wheel(0.)

ryb = [ryb_to_rgb(1.,0.,0.),ryb_to_rgb(1.,0.5,0.),ryb_to_rgb(1.,1.,0.),ryb_to_rgb(0.5,1.,0.),\
	   ryb_to_rgb(0.,1.,0.),ryb_to_rgb(0.,1.,0.5),ryb_to_rgb(0.,1.,1.),ryb_to_rgb(0.,0.5,1.),\
	   ryb_to_rgb(0.,0.,1.),ryb_to_rgb(0.5,0.,1.),ryb_to_rgb(1.,0.,1.),ryb_to_rgb(1.,0.,0.5)]

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
			 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
			 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
			 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
			 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableau20)):    
	tableau20[i] = norm_rgb(tableau20[i])

# subsets of tableau20
tableau20_solid = tableau20[::2]
tableau20_light = tableau20[1::2]
# reordering of tableau20 with all dark then all light colours
tableau20_alt = tableau20_solid + tableau20_light
