from functools import partial
import torchvision
from torchvision.utils import draw_bounding_boxes
# Import the distinctipy module
from distinctipy import distinctipy
from cjm_psl_utils.core import download_file, file_extract

from inspect import class_names


# Visualizing Image Annotations
# Generate a color map
# Generate a list of colors with a length equal to the number of labels
colors = distinctipy.get_colors(len(class_names))

# Make a copy of the color map in integer format
int_colors = [tuple(int(c*255) for c in color) for color in colors]

# Generate a color swatch to visualize the color map
distinctipy.color_swatch(colors)


# Download a font file
# Set the name of the font file
font_file = 'KFOlCnqEu92Fr1MmEU9vAw.ttf'

# Download the font file
download_file(f"https://fonts.gstatic.com/s/roboto/v30/{font_file}", "./")

# Define the bounding box annotation function
draw_bboxes = partial(draw_bounding_boxes, fill=True, width=4, font=font_file, font_size=25)