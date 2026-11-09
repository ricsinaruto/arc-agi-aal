"""Transcript -> turns. Each turn: {step, action, time, user, results:[...], thinking chars, ...}."""
import re

HEAD = re.compile(r"(?m)^--- analysis_step=(\d+) \| action=(\d+) \| (\d\d:\d\d:\d\d) \| tool-agent ---$")
SECT = re.compile(r"(?m)^\[(SYSTEM PROMPT|USER PROMPT|MODEL RESPONSE META|THINKING|TOOL CALL: python|TOOL RESULT: python|ASSISTANT|ANALYZER STATUS)\]$")


def turns(path):
    s = open(path, errors="replace").read()
    heads = list(HEAD.finditer(s))
    out = []
    for i, h in enumerate(heads):
        body = s[h.end():heads[i + 1].start() if i + 1 < len(heads) else len(s)]
        marks = list(SECT.finditer(body))
        sections = []
        for j, m in enumerate(marks):
            sections.append((m.group(1), body[m.end():marks[j + 1].start() if j + 1 < len(marks) else len(body)].strip("\n")))
        out.append({"step": int(h.group(1)), "action": int(h.group(2)), "time": h.group(3), "sections": sections})
    return out


def norm(line):
    line = re.sub(r"\d+", "N", line)
    return " ".join(line.split()[:7])
