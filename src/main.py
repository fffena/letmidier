import argparse
from textwrap import dedent
from fontTools.ttLib import TTFont
from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.svgPathPen import SVGPathPen

import exceptions as exp
import utils


class Cli:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.err = None
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
        if self.err:
            raise self.err
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
        glyphes = font.getGlyphSet()
        ascender: int = font["hhea"].ascender  # type: ignore
        descender: int = font["hhea"].descender  # minus value

        tg_glyphs = [glyphes[cmap.get(ord(i))] for i in args.text]
        aiueo = []
        for i in tg_glyphs:
            pen = RecordingPen()
            svgpen = SVGPathPen(glyphes)
            i.draw(pen)
            i.draw(svgpen)
            aiueo.append((i.width, i.height))
            with open(f"../aseets/{i.name}.svg", "w") as f:
                svg = dedent(f"""
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0, 0, {i.width}, {i.height}">
                    <g transform="translate(0, {ascender+descender}) scale(1, -1)">
                        <path d="{svgpen.getCommands()}" />
                    </g>
                </svg>
                """).strip("\n")
                f.write(svg)
            original = utils.svg2pil(svg)
            resized = original.resize((50, round(original.height * 50 / original.width)))
            resized.save(f"../aseets/{i.name}.png")
        print(aiueo)
        print("Hello, World!", args.text)


if __name__ == "__main__":
    Cli().run_cmd()
