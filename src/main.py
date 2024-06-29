import argparse


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
            help="文字の作成で使用されるフォント。URLを入力することもできます。"
        )

    def run_cmd(self):
        return self.args.func(self.args)

    def create_midi(self, args):
        print("Hello, World!", args.text)


if __name__ == "__main__":
    Cli().run_cmd()
