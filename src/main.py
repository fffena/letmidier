import argparse
from fontTools.ttLib import TTFont
from fontTools.pens.recordingPen import RecordingPen

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
        font_arg = create_midi.add_argument(
            "-f", "--font-name",
            help="文字の作成で使用されるフォント。URLを入力することもできます。",
        )
        try:
            font_arg.default = utils.get_system_font().pop()
        except exp.CannotFindSystemFont:
            self.err = exp.CannotFindSystemFont(
                "システムのデフォルトフォントを探すことができませんでした。-fオプションでフォントを直接指定してください。"
            )

    def run_cmd(self):
        if self.err:
            raise self.err
        if hasattr(self.args, "func"):
            return self.args.func(self.args)
        else:
            self.parser.print_help()

    def create_midi(self, args):
        font = TTFont(args.font_name)
        cmap = font.getBestCmap()
        glyphes = font.getGlyphSet()

        tg_glyph = glyphes[cmap[ord(args.text)]]
        print("Hello, World!", args.text)


if __name__ == "__main__":
    Cli().run_cmd()
