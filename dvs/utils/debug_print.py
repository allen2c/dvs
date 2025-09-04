import typing

import rich.box
import rich.console
import rich.table
import rich.text
from rich_color_support import RichColorRotator, RichColorsBase


def debug_print(
    text: typing.Any,
    *,
    title: str = "Debug Print",
    footer: str = "",
    console: rich.console.Console | None = None,
    verbose: bool | None = False,
    rich_color: RichColorsBase | str | None = None,
    width: int = 80,
):
    if not verbose:
        return
    if console is None:
        console = rich.console.Console()
    if rich_color is None:
        rich_color = RichColorRotator().pick()

    table = rich.table.Table(
        box=rich.box.HORIZONTALS, width=width, show_footer=True if footer else False
    )
    table.add_column(title, footer=footer)
    table.add_row(rich.text.Text(str(text)[:4000]))
    console.print(table, style=rich_color)

    return None


if __name__ == "__main__":
    debug_print("Hello, world!" * 80, title="Debug Print", verbose=True)
