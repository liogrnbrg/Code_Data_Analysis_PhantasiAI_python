import re


def make_safe_filename(name):
    name = str(name)
    return re.sub(r"[^\w\-]+", "_", name)


def make_display_label(name):
    name = str(name)
    return name.replace("_", " ")