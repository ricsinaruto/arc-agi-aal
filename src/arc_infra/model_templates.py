"""Request template controls. Only Flash is served (rl-min: the other checkpoints' controls were removed), so requests
pass through unchanged."""


def text_only(model):
    return False


def request_messages(model, messages):
    return messages


def allowed_efforts(model):
    return ("xhigh", "medium", "low")


def template_kwargs(model, kwargs):
    return dict(kwargs)
