"""Engine watchdog patches (anchor edits; the same anchors in the v8.2 / v8.5b / AV2C bundle trees)."""

BOOTSTRAP = ("bootstrap_arena.py", '''            command = [
                python, "-m", "arena.run", "--source"''', '''            if study.get("engine_watchdog"):
                # Engine watchdog (17 Oct): the trial restarts a dead server with this command (arena.engine_watchdog).
                env["ARENA_ENGINE_WATCHDOG"] = json.dumps({
                    "start": [*serving_command, "--working", str(working / "serving"), "--input-root", args.input_root],
                    "stop": [python, "-c", stop_program, str(working / "serving"), args.input_root],
                    "env": current_env or {},
                    "test_kill_seconds": (study["engine_watchdog"] if isinstance(study["engine_watchdog"], dict)
                                          else {}).get("test_kill_seconds")})
            command = [
                python, "-m", "arena.run", "--source"''')

TRIAL = ("src/arena/trial.py", "        scheduler.start_all()\n", '''        scheduler.start_all()
        from . import engine_watchdog as _engine_watchdog
        _engine_watchdog.start(self.base_url, self.server_dir)
''')

DUCK = ("src/arena/duck.py", '''        for attempt, delay in enumerate((*SERVER_ERROR_DELAYS, None), 1):
            response = await call(payload)
            if (response.status_code < 500 or delay is None or self.stopping.is_set()
                    or self.remaining() <= delay + 1):
                return response''', '''        from . import engine_watchdog as _wd

        if _wd.enabled():
            # Engine watchdog (17 Oct): a failure waits for the watchdog (one poll, then any restart) instead of the
            # fixed backoff; a restart since the request began resets the attempt count.
            attempt, restarts = 0, _wd.STATE["restarts"]
            while True:
                attempt += 1
                try:
                    response, error = await call(payload), None
                except httpx.TransportError as exc:
                    response, error = None, exc
                if response is not None and response.status_code < 500:
                    return response
                if self.stopping.is_set() or self.remaining() <= 30:
                    if error is not None:
                        raise error
                    return response
                if _wd.STATE["restarts"] != restarts:
                    attempt, restarts = 1, _wd.STATE["restarts"]
                if attempt > len(SERVER_ERROR_DELAYS) and _wd.HEALTHY.is_set():
                    if error is not None:
                        raise error
                    return response
                self.server_error_retries += 1
                self.workspace.append("server_error_retry", step=self.steps, attempt=attempt, delay="watchdog",
                                      status=None if response is None else response.status_code,
                                      detail=repr(error)[:300] if error is not None else response.text[:300])
                await _wd.settle(self.remaining() - 30)
        for attempt, delay in enumerate((*SERVER_ERROR_DELAYS, None), 1):
            response = await call(payload)
            if (response.status_code < 500 or delay is None or self.stopping.is_set()
                    or self.remaining() <= delay + 1):
                return response''')

PATCHES = (BOOTSTRAP, TRIAL, DUCK)
