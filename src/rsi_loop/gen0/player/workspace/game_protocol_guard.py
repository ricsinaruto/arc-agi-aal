#!/usr/bin/env python3
"""Root-owned phase guard that forwards only valid bootstrap operations to one broker."""
from __future__ import annotations
import argparse, contextlib, json, os, signal, socket, stat, subprocess, sys, time
from pathlib import Path
MAX_REQUEST_LINE=1<<20
MAX_RESPONSE_LINE=16<<20
STATUS_FORWARD_TIMEOUT_SECONDS=5.0

class GuardError(RuntimeError): pass

def _peer_credentials(connection: socket.socket) -> tuple[int, int, int]:
    if not hasattr(socket,"SO_PEERCRED"): raise GuardError("SO_PEERCRED unavailable")
    import struct
    return struct.unpack("3i",connection.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,struct.calcsize("3i")))

def _is_descendant(pid: int, ancestor: int) -> bool:
    if pid<=0 or ancestor<=0: return False
    for _ in range(64):
        if pid==ancestor: return True
        try: tail=Path(f"/proc/{pid}/stat").read_text().rsplit(") ",1)[1].split();pid=int(tail[1])
        except (OSError,ValueError,IndexError): return False
        if pid<=1: return False
    return False

def _write_prewarm_ready(path: Path, *, listen: Path, peer_uid: int, peer_gid: int,
                         peer_pid: int, child_pid: int, release_fd: int) -> None:
    release=os.fstat(release_fd)
    payload=json.dumps({"schema":"arc3-process-prewarm-ready/v1","guard_socket":str(listen),
                        "peer_uid":peer_uid,"peer_gid":peer_gid,"peer_pid":peer_pid,
                        "guard_pid":os.getpid(),"child_pid":child_pid,
                        "release_device":release.st_dev,"release_inode":release.st_ino},sort_keys=True,separators=(",",":"))
    flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC
    if hasattr(os,"O_NOFOLLOW"): flags|=os.O_NOFOLLOW
    fd=os.open(path,flags,0o600)
    try:
        data=(payload+"\n").encode();written=0
        while written<len(data): written+=os.write(fd,data[written:])
        os.fsync(fd)
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid!=0 or info.st_gid!=0 or (info.st_mode&0o777)!=0o600:
            raise GuardError("prewarm readiness receipt identity is invalid")
    finally: os.close(fd)
    directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_CLOEXEC|os.O_NOFOLLOW)
    try: os.fsync(directory)
    finally: os.close(directory)

def _line(connection: socket.socket, *, limit: int = MAX_REQUEST_LINE, kind: str = "request") -> bytes:
    if limit <= 0: raise GuardError("invalid frame bound")
    data=b""
    while not data.endswith(b"\n"):
        chunk=connection.recv(min(65536,limit+1-len(data)))
        if not chunk: break
        data+=chunk
        if len(data)>limit: raise GuardError(f"{kind} exceeds bound")
    if not data.endswith(b"\n"): raise GuardError(f"{kind} must be one newline-terminated frame")
    return data

def _single_genuine(request: object) -> bool:
    if not isinstance(request,dict) or set(request)!={"op","actions"} or request.get("op")!="act": return False
    actions=request.get("actions")
    return (isinstance(actions,list) and len(actions)==1 and isinstance(actions[0],dict) and
            set(actions[0])=={"name","data"} and isinstance(actions[0].get("name"),str) and
            actions[0]["name"]!="RESET" and isinstance(actions[0].get("data"),dict))

def _allowed(phase: int, request: object) -> bool:
    if phase==0: return isinstance(request,dict) and request=={"op":"observe"}
    if phase in {1,2}: return _single_genuine(request)
    return (isinstance(request,dict) and request.get("op") in {"observe","status","act"})

def _trusted_socket(path: Path) -> None:
    info=os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISSOCK(info.st_mode) or info.st_uid!=0 or info.st_mode & 0o007:
        raise GuardError("upstream broker socket is not root-owned and private")

def _open_prewarm_release(path: Path) -> int:
    if not path.is_absolute() or path.name!="prewarm.release": raise GuardError("invalid prewarm release path")
    before=os.lstat(path)
    if (stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode) or before.st_uid!=0 or
            before.st_gid!=0 or (before.st_mode&0o777)!=0o644 or before.st_size!=0):
        raise GuardError("prewarm release must be fresh root-owned mode-0644 empty regular file")
    fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
    opened=os.fstat(fd)
    if (opened.st_dev,opened.st_ino)!=(before.st_dev,before.st_ino): os.close(fd);raise GuardError("prewarm release changed while opened")
    return fd

def _require_prewarm_release(fd: int) -> None:
    info=os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid!=0 or info.st_gid!=0 or
            (info.st_mode&0o777)!=0o644 or info.st_size!=3 or os.pread(fd,4,0)!=b"GO\n"):
        raise GuardError("root launcher has not released process prewarm")

def _forward(upstream: Path, frame: bytes, *, status_only: bool = False) -> tuple[bytes,dict]:
    _trusted_socket(upstream)
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as connection:
        if status_only: connection.settimeout(STATUS_FORWARD_TIMEOUT_SECONDS)
        connection.connect(str(upstream));connection.sendall(frame);reply=_line(connection,limit=MAX_RESPONSE_LINE,kind="response")
    value=json.loads(reply)
    if not isinstance(value,dict) or not isinstance(value.get("ok"),bool): raise GuardError("invalid broker reply")
    fatal=value.get("fatal",False)
    if not isinstance(fatal,bool) or (fatal and value.get("ok")): raise GuardError("invalid broker fatal marker")
    return reply,value

def serve(listen: Path, upstream: Path, ready: Path, release: Path, peer_uid: int, peer_gid: int, command: list[str]) -> int:
    if os.geteuid()!=0 or peer_uid<100000 or peer_gid<100000 or not command: raise GuardError("invalid trusted guard invocation")
    synthetic=os.environ.get("ARC3_SYNTHETIC_REHEARSAL")=="1"
    canonical_pair=(listen.parent==upstream.parent and listen.name[:-6]==upstream.name[:-6] and upstream.name.endswith("b.sock"))
    if (not listen.is_absolute() or not upstream.is_absolute() or listen.parent.parent!=Path("/run/arc3o") or
            not listen.name.endswith("g.sock") or (not synthetic and not canonical_pair)):
        raise GuardError("guard and broker socket capabilities are not canonically paired")
    expected_ready=listen.with_name(listen.name[:-6]+"p.ready")
    if ready!=expected_ready: raise GuardError("prewarm readiness path is not canonically paired")
    for directory in (listen.parent.parent,listen.parent):
        parent=os.lstat(directory)
        if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode) or parent.st_uid!=0 or parent.st_mode & 0o022:
            raise GuardError("socket namespace is not root-controlled")
    # The official broker socket deliberately does not exist during process
    # prewarm. Every gameplay request revalidates it immediately in _forward.
    if upstream.exists() or upstream.is_symlink(): _trusted_socket(upstream)
    if listen.exists() or listen.is_symlink(): raise GuardError("guard socket already exists")
    if ready.exists() or ready.is_symlink(): raise GuardError("prewarm readiness receipt already exists")
    release_fd=_open_prewarm_release(release)
    child=None
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
        server.bind(str(listen));os.chown(listen,peer_uid,peer_gid);os.chmod(listen,0o600);server.listen(8);server.settimeout(.2)
        created=os.lstat(listen);created_identity=(created.st_dev,created.st_ino)
        child=subprocess.Popen(command,user=peer_uid,group=peer_gid,extra_groups=[])
        stop_requested={"value":False}
        def request_child_stop(_signum, _frame):
            stop_requested["value"]=True
            if child.poll() is None: child.terminate()
        for signum in (signal.SIGINT,signal.SIGTERM): signal.signal(signum,request_child_stop)
        phase=0;prewarm_ready=False
        try:
            while child.poll() is None:
                try: connection,_=server.accept()
                except socket.timeout: continue
                with connection:
                    connection_pid,connection_uid,connection_gid=_peer_credentials(connection)
                    if (connection_uid!=peer_uid or connection_gid!=peer_gid or
                            not _is_descendant(connection_pid,child.pid)): continue
                    try: frame=_line(connection);request=json.loads(frame)
                    except (GuardError,ValueError,TypeError,json.JSONDecodeError) as exc:
                        with contextlib.suppress(OSError):
                            connection.sendall(json.dumps({"ok":False,"error":"ProtocolGuardError","message":str(exc)},separators=(",",":")).encode()+b"\n")
                        continue
                    if request=={"op":"prewarm_ready"}:
                        if prewarm_ready or phase!=0:
                            connection.sendall(b'{"ok":false,"error":"ProtocolGuardError","message":"prewarm readiness is one-use"}\n');continue
                        _write_prewarm_ready(ready,listen=listen,peer_uid=peer_uid,peer_gid=peer_gid,peer_pid=connection_pid,child_pid=child.pid,release_fd=release_fd)
                        prewarm_ready=True
                        connection.sendall(b'{"ok":true,"value":{"prewarm":"READY"}}\n');continue
                    if not prewarm_ready:
                        connection.sendall(b'{"ok":false,"error":"ProtocolGuardError","message":"prewarm readiness required"}\n');continue
                    _require_prewarm_release(release_fd)
                    if not _allowed(phase,request):
                        response={"ok":False,"error":"ProtocolGuardError","message":"immediate model-selected bootstrap action required"}
                        connection.sendall(json.dumps(response,separators=(",",":")).encode()+b"\n");continue
                    # Upstream failures are fatal. Never continue after ambiguity
                    # about whether an official operation was acknowledged.
                    try:
                        reply,value=_forward(upstream,frame,status_only=request=={"op":"status"})
                    except (TimeoutError,socket.timeout) as exc:
                        if request!={"op":"status"}: raise
                        response={"ok":False,"error":"ProtocolGuardError","message":"broker status timed out"}
                        with contextlib.suppress(OSError):
                            connection.sendall(json.dumps(response,separators=(",",":")).encode()+b"\n")
                        continue
                    if value.get("ok"):
                        if phase==0: phase=1
                        elif phase==1: phase=2 if isinstance(value.get("value"),dict) and value["value"].get("terminal")=="ACTIVE" else 3
                        elif phase==2: phase=3
                    connection.sendall(reply)
                    if value.get("fatal"):
                        # The broker has permanently revoked this game's action
                        # capability. End the retained Prime process now rather
                        # than spending provider requests against a dead slot.
                        if child.poll() is None: child.terminate()
                        try: child.wait(10)
                        except subprocess.TimeoutExpired: child.kill();child.wait(5)
                        return 1
            return 0 if stop_requested["value"] else (child.returncode or 0)
        finally:
            os.close(release_fd)
            if child.poll() is None:
                child.terminate()
                try: child.wait(10)
                except subprocess.TimeoutExpired: child.kill();child.wait(5)
            try: current=os.lstat(listen)
            except FileNotFoundError: pass
            else:
                if (current.st_dev,current.st_ino)==created_identity: listen.unlink()

def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--listen",type=Path,required=True);parser.add_argument("--upstream",type=Path,required=True);parser.add_argument("--prewarm-ready",type=Path,required=True);parser.add_argument("--prewarm-release",type=Path,required=True);parser.add_argument("--peer-uid",type=int,required=True);parser.add_argument("--peer-gid",type=int,required=True);parser.add_argument("command",nargs=argparse.REMAINDER)
    args=parser.parse_args();command=args.command[1:] if args.command[:1]==["--"] else args.command
    try:return serve(args.listen,args.upstream,args.prewarm_ready,args.prewarm_release,args.peer_uid,args.peer_gid,command)
    except (GuardError,OSError) as exc: raise SystemExit(f"protocol guard refused: {exc}") from exc
if __name__=="__main__": raise SystemExit(main())
