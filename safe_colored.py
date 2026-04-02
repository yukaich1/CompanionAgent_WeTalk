class _EmptyColor:
    def __getattr__(self, _name):
        return ""


try:
    from colored import Fore as _Fore, Style as _Style

    Fore = _Fore
    Style = _Style
except Exception:
    Fore = _EmptyColor()
    Style = _EmptyColor()
