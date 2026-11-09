"""Client for exactly one fixed independent ARC broker endpoint."""
from __future__ import annotations
import json, os, socket
from pathlib import Path
SOCKET=Path(os.environ["ARC3_GAME_BROKER_SOCKET"])
# Status is read-only and must never strand a terminal gate or recovery probe.
# Mutating act calls deliberately have no client-side timeout: their host broker
# owns the longer action deadline and ambiguity policy.
STATUS_TIMEOUT_SECONDS=5.0
_STATE_PATH=Path(os.environ["ARC3_STATE"])/"fixed-client-bootstrap-phase" if os.environ.get("ARC3_STATE") else None
_BOOTSTRAP_PHASE=0

def _phase():
    if _STATE_PATH is None or not _STATE_PATH.exists(): return _BOOTSTRAP_PHASE
    try: value=int(_STATE_PATH.read_text().strip())
    except (OSError,ValueError): return 0
    return value if value in {0,1,2,3} else 0

def _set_phase(value):
    global _BOOTSTRAP_PHASE
    _BOOTSTRAP_PHASE=value
    if _STATE_PATH is not None:
        temporary=_STATE_PATH.with_name(_STATE_PATH.name+".tmp")
        temporary.write_text(str(value)+"\n");os.chmod(temporary,0o600);os.replace(temporary,_STATE_PATH)

def _single_genuine(actions):
    return (isinstance(actions,list) and len(actions)==1 and isinstance(actions[0],dict) and
            set(actions[0])=={"name","data"} and isinstance(actions[0].get("name"),str) and
            actions[0]["name"] != "RESET" and isinstance(actions[0].get("data"),dict))

def call(op:str, **kw):
    allowed={"observe":set(),"status":set(),"act":{"actions"}}
    if op not in allowed or set(kw)!=allowed[op]: raise ValueError("fixed broker operation refused")
    phase=_phase()
    if phase==0 and op!="observe":
        raise RuntimeError("first game operation must be observe")
    if phase in {1,2} and (op!="act" or not _single_genuine(kw.get("actions"))):
        raise RuntimeError("immediate single genuine non-RESET action required before another game operation")
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as s:
        if op == "status": s.settimeout(STATUS_TIMEOUT_SECONDS)
        s.connect(str(SOCKET));s.sendall(json.dumps({"op":op,**kw},separators=(",",":")).encode()+b"\n");raw=b""
        while not raw.endswith(b"\n"):
            x=s.recv(1<<20)
            if not x:raise RuntimeError("broker disconnected")
            raw+=x
    r=json.loads(raw)
    if not r.get("ok"):raise RuntimeError(r.get("error","BrokerError")+": "+r.get("message",""))
    value=r["value"]
    if phase==0:
        _set_phase(1)
    elif phase==1:
        _set_phase(2 if isinstance(value,dict) and value.get("terminal")=="ACTIVE" else 3)
    elif phase==2:
        _set_phase(3)
    return value
def observe():return call("observe")
def status():return call("status")
def act(actions):return call("act",actions=actions)

def _main():
    import argparse
    parser=argparse.ArgumentParser(description="fixed one-game ARC broker client")
    parser.add_argument("operation", choices=("observe","status","act","terminal-gate"))
    parser.add_argument("--actions")
    args=parser.parse_args()
    if args.operation=="act":
        if args.actions is None: parser.error("act requires --actions JSON")
        try: actions=json.loads(args.actions)
        except json.JSONDecodeError: parser.error("--actions must be valid JSON")
        value=act(actions)
    else:
        if args.actions is not None: parser.error("--actions is valid only for act")
        value=observe() if args.operation=="observe" else status()
    if args.operation=="terminal-gate":
        terminal=value.get("terminal") if isinstance(value,dict) else None
        if terminal in {"WIN","ACTION_CAP"}: return
        raise SystemExit("terminal gate pending")
    print(json.dumps(value,sort_keys=True,separators=(",",":")))

if __name__=="__main__":
    _main()
