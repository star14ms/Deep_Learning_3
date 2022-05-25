from console import Console
from rich.progress import track, Progress, SpinnerColumn, TimeElapsedColumn
import time

# for step in track(range(100)):
#     time.sleep(1)


n = 0

console = Console(record=True)
with Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    TimeElapsedColumn(),
    console=console,
    transient=False,
) as progress:
    task1 = progress.add_task("[red]Downloading", total=1000)
    task2 = progress.add_task("[green]Processing", total=1000)
    task3 = progress.add_task("[yellow]Thinking", total=None)
    while not progress.finished:
        progress.update(task1, advance=0.5)
        progress.update(task2, advance=0.3)
        time.sleep(0.5)
        progress.log(n)
        n += 1
