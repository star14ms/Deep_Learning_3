from rich_console import Console
from rich.progress_bar import ProgressBar


if __name__ == "__main__":  # pragma: no cover
    console = Console()
    bar = ProgressBar(width=50, total=100)

    import time

    console.show_cursor(False)
    for n in range(0, 101, 1):
        bar.update(n)
        console.print(bar)
        console.file.write("\r")
        time.sleep(0.05)
    console.show_cursor(True)
    console.print()