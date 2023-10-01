import os
from PIL import Image
import utils
import converter
import threading
import config

image_colorizer = converter.Converter()

def image_converter(image):
    global image_colorizer
    image = Image.fromarray(image_colorizer.convert(image))
    output_path = f'{config.output_path}/{image}'
    image.save(output_path)

def main():
    files = os.listdir(config.input_path)
    image_files = []

    for file in files:
        if os.path.isfile(file) and utils.is_image_file(file):
            image_files.append(file)

    for file in image_files:
        threading.Thread(target=image_converter, args=file).start()

    print("done...")

if __name__ == "__main__":
    main()
