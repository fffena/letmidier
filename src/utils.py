from find_system_fonts_filename import (
    get_system_fonts_filename,
    FindSystemFontsFilenameException,
)

import exceptions as exp


def get_system_font():
    try:
        fonts = get_system_fonts_filename()
    except FindSystemFontsFilenameException:
        raise exp.CannotFindSystemFont("Cannot find system font.")
    # TODO
    return fonts
