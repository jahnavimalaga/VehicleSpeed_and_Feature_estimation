#Github link: https://github.com/MarvinKweyu/ColorDetect
import sys

from colordetect import ColorDetect

#path to the image in the command line argument
user_image = ColorDetect(sys.argv[1])
# return dictionary of color count. Do anything with this
#Sample output: How much percentage of the image consits of which color
print("output:",user_image.get_color_count())

# write color count
#user_image.write_color_count()
# optionally, write any text to the image
#user_image.write_text(text="hellow",font_thickness=1)

# save the image after using either of the options (write_color_count/write_text) or both
#user_image.save_image("results","outputImage.png")