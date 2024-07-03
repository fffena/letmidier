from io import BytesIO

from cairosvg import svg2png
from find_system_fonts_filename import (
    get_system_fonts_filename,
    FindSystemFontsFilenameException,
)
from PIL import Image

import exceptions as exp


def get_installed_font():
    try:
        fonts = get_system_fonts_filename()
    except FindSystemFontsFilenameException:
        raise exp.CannotFindSystemFont("Cannot find system font.")
    # TODO
    return fonts


def svg2pil(svg: str, dpi: int = 72):
    dist = BytesIO()
    svg2png(bytestring=svg, background_color="white", dpi=dpi, write_to=dist)
    return Image.open(dist)
