from io import BytesIO

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from find_system_fonts_filename import (
    get_system_fonts_filename,
    FindSystemFontsFilenameException,
)

import exceptions as exp


def get_installed_font():
    try:
        fonts = get_system_fonts_filename()
    except FindSystemFontsFilenameException:
        raise exp.CannotFindSystemFont("Cannot find system font.")
    # TODO
    return fonts


def svg2pil(svg: str, dpi: int = 72):
    draw_data = svg2rlg(BytesIO(svg.encode()))
    return renderPM.drawToPIL(draw_data, dpi=dpi)
