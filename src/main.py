import argparse
from textwrap import dedent

from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont

import exceptions as exp
import utils


class Cli:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self._define_args()
        self.args = self.parser.parse_args()

    def _define_args(self):
        subparsers = self.parser.add_subparsers()

        create_midi = subparsers.add_parser("create")
        create_midi.set_defaults(func=self.create_midi)
        create_midi.add_argument("text", help="入力するテキスト")
        create_midi.add_argument(
            "-f", "--font-name",
            help="文字の作成で使用されるフォント。URLを入力することもできます。",
        )

    def run_cmd(self):
        if hasattr(self.args, "func"):
            return self.args.func(self.args)
        else:
            self.parser.print_help()

    def create_midi(self, args):
        try:
            if args.font_name is None:
                font = utils.get_installed_font().pop()
            else:
                font = args.font_name
        except exp.CannotFindSystemFont:
            raise exp.CannotFindSystemFont(
                "システムのデフォルトフォントを探すことができませんでした。-fオプションでフォントを直接指定してください。"
            )

        font = TTFont(font)
        cmap = font.getBestCmap()
        glyphs = font.getGlyphSet()
        ascender: int = font["hhea"].ascender  # type: ignore
        descender: int = font["hhea"].descender  # minus value
        line_gap = font["hhea"].lineGap

        tg_glyphs = [glyphs[cmap.get(ord(i))] for i in args.text]
        aiueo = []
        for i in tg_glyphs:
            pen = RecordingPen()
            svg_pen = SVGPathPen(glyphs)
            i.draw(pen)
            i.draw(svg_pen)
            height = i.height or ascender - descender
            aiueo.append((i.width, height))
            with open(f"../assets/{i.name}.svg", "w") as f:
                svg = dedent(f"""
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0, 0, {i.width}, {height}">
                    <g transform="translate(0, {height-line_gap-21}) scale(1, -1)">
                        <path d="{svg_pen.getCommands()}" />
                    </g>
                </svg>
                """).strip("\n")
                f.write(svg)
            original = utils.svg2pil(svg)
            resized = original.resize((round(original.width * 100 / original.height), 100))
            resized.save(f"../assets/{i.name}.png")
        print(aiueo)
        print("Hello, World!", args.text)


if __name__ == "__main__":
    Cli().run_cmd()
