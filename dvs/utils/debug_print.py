import rich.box
import rich.console
import rich.table
import rich.text
from rich_color_support import RichColorRotator, RichColorsBase


def debug_print(
    text: str,
    *,
    title: str = "Debug Print",
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

    table = rich.table.Table(box=rich.box.HORIZONTALS, width=width)
    table.add_column(title)
    table.add_row(rich.text.Text(str(text)))
    console.print(table, style=rich_color)

    return None


if __name__ == "__main__":
    debug_print("Hello, world!" * 80, title="Debug Print", verbose=True)
