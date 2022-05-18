from rich import print

print("Hello, [bold magenta]World[/bold magenta]!", ":vampire:", locals())


from rich_console import Console
console = Console()
# try:
#     raise
# except Exception:
#     console.print_exception(show_locals=False)


def foo(n):
    return bar(n)


def bar(n):
    return foo(n)


console = Console()

try:
    foo(1)
except Exception:
    console.print_exception(max_frames=10)