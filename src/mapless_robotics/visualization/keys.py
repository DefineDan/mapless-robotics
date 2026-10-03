"""Key names and movement mapping shared by visualization backends."""

UP = {63232, 65362, 2490368}
DOWN = {63233, 65364, 2621440}
LEFT = {63234, 65361, 2424832}
RIGHT = {63235, 65363, 2555904}


def key_name(code):
    if code in UP:
        return "up"
    if code in DOWN:
        return "down"
    if code in LEFT:
        return "left"
    if code in RIGHT:
        return "right"
    if code in (ord("q"), 27):
        return "quit"
    if code in (ord("f"), ord("g")):
        return chr(code)
    if code in (ord("a"), ord("d"), ord("w"), ord("s")):
        return chr(code)
    return None


def action_for_key(key):
    return {
        "up": "move_forward",
        "down": "move_backward",
        "left": "move_left",
        "right": "move_right",
        "a": "turn_left",
        "d": "turn_right",
        "w": "look_up",
        "s": "look_down",
    }.get(key)
