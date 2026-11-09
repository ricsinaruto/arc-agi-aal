"""Competition gateway and public rehearsal lifecycle for the arena agent.

The public rehearsal uses the SDK's competition-mode REST API and unique cyclic
clones. A real rerun enumerates only the gateway; it never loads public games.
"""
import copy
import json
import logging
import os
import time
from contextlib import contextmanager

from arc_agi import Arcade, OperationMode


class ScorecardOwner:
    """Close the one scorecard even if worker initialization fails.

    An uncertain close is not replayed. The underlying SDK owns action transport;
    this wrapper does not retry or intercept gameplay actions.
    """

    def __init__(self, arcade, output):
        self.arcade, self.output = arcade, output
        self.card = None
        self.open_attempted = self.close_attempted = False

    def __getattr__(self, key):
        return getattr(self.arcade, key)

    def open_scorecard(self, *args, **kwargs):
        if self.open_attempted:
            raise RuntimeError('Competition allows one scorecard; refusing another open')
        self.open_attempted = True
        self.card = self.arcade.open_scorecard(*args, **kwargs)
        if not self.card:
            raise RuntimeError('Competition scorecard did not open')
        return self.card

    def close_scorecard(self, card):
        if card != self.card or self.close_attempted:
            raise RuntimeError('Invalid or repeated competition scorecard close')
        self.close_attempted = True
        closed = self.arcade.close_scorecard(card)
        if closed is not None:
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output / 'gateway-scorecard.json').write_text(
                json.dumps(closed.model_dump(mode='json', exclude={'api_key'}), indent=2))
        return closed

    def finalize(self):
        if self.card is not None and not self.close_attempted:
            self.close_scorecard(self.card)

    # Set by open_arcade to the local research gateway (public cyclic clones) only; never on the hidden rerun.
    local_gateway = None

    def peek_score(self, card):
        """Live progress (hero-v2, 29 September): the open scorecard's scores, read in-process from the local research
        gateway (its REST API refuses reads of a competition-mode card). Monitoring only: nothing reaches an agent or
        the scheduler. None on the hidden gateway or when the card is not ours."""
        server = self.local_gateway
        if server is None or card != self.card or getattr(server, '_arcade', None) is None:
            return None
        from arc_agi.scorecard import EnvironmentScorecard

        arcade = server._arcade
        for _ in range(3):  # the gateway's request threads may add a card while we iterate
            try:
                scorecard = arcade.scorecard_manager.scorecards.get(card)
                if scorecard is None:
                    return None
                return EnvironmentScorecard.from_scorecard(scorecard, arcade.available_environments)
            except RuntimeError:
                time.sleep(0.05)
        return None


def is_rerun():
    return os.environ.get('KAGGLE_IS_COMPETITION_RERUN', '').lower() in ('1', 'true')


def is_rehearsal(study):
    """Rerun branch forced in an ordinary kernel; only the gateway host and roster differ."""
    return is_rerun() and bool((study.get('competition') or {}).get('rehearsal'))


def runtime_seconds(study):
    settings = study['competition']
    return settings['submission_seconds'] if is_rerun() else study['notebook_seconds']


def gameplay_seconds(study, hard_end, *, now=None):
    now = time.time() if now is None else now
    available = hard_end - now - study['competition']['cleanup_seconds']
    if available <= 0:
        raise TimeoutError('No competition gameplay budget remains after setup')
    if is_rerun():
        return available
    requested = study['competition']['preview_seconds']
    # An unconditional run stop (arena.early_stop ``always``; hero-v2-blocks): the preview wall stays the study's (the
    # scheduler and the agents see it) but only at_seconds of it are played, so only those must fit.
    stop = study.get('early_stop') or {}
    needed = min(requested, stop['at_seconds']) if stop.get('always') else requested
    if available < needed:
        raise TimeoutError('Insufficient time for the complete public rehearsal')
    return requested


def resolve_blocks(resolved, notebook_start, *, now=None):
    """The blocks scheduler on the competition gateway (the blocks3-plans-prune25-40 submission).

    The scheduler's wall (block length = wall / blocks) is the play time the submission really has: the
    ``submission_seconds`` envelope from the notebook start, less the setup measured so far and the cleanup reserve.
    On the hidden rerun that is exactly the trial wall ``gameplay_seconds`` resolved. The public preview projects the
    same wall (its setup is the rerun's) and plays only ``preview_seconds`` of it (an unconditional stop at the preview
    length), so the preview is block 1 of the hidden run as the agents and the scheduler will see it. The roster size is
    the gateway's: a streams partition that does not split into ``blocks`` groups of ``slots`` takes as many blocks as
    it needs (the stream count, the partition and the fill rule are unchanged). Mutates and returns ``resolved``.
    """
    if resolved.get('blocks') is None or (resolved.get('competition') or {}).get('research_clones'):
        # Research-clone studies (the nine-hour Modal campaign: 110 public clones, submission = preview = the study
        # wall) keep the build's blocks wall and plan, as every campaign bundle before this function did; the gateway
        # projection would refuse them (a wall shorter than the preview after setup).
        return resolved
    import math

    from .blocks import settings as blocks_settings
    from .early_stop import settings as early_stop_settings

    now = time.time() if now is None else now
    competition = resolved['competition']
    envelope_end = float(notebook_start) + float(competition['submission_seconds'])
    play = envelope_end - now - float(competition['cleanup_seconds'])
    if is_rerun():
        wall = float(resolved['wall_seconds'])  # gameplay_seconds: the hidden rerun's envelope less cleanup
    else:
        wall = play
        preview = int(competition['preview_seconds'])
        if wall <= preview:
            raise TimeoutError('The projected competition wall is not longer than the public preview')
        resolved['wall_seconds'] = wall
        resolved['early_stop'] = early_stop_settings({'at_seconds': preview, 'always': True}, wall)
    blocks = dict(resolved['blocks'])
    nominal = dict(blocks)
    games = len(resolved['games'])
    blocks.update(wall_seconds=wall, roster_games=games)
    if blocks.get('partition') == 'streams':
        slots = int(blocks['slots'])
        last = games - slots * (int(blocks['blocks']) - 1)
        if not 1 <= last <= slots:
            blocks['blocks'] = max(1, math.ceil(games / slots))
    cfg = blocks_settings(blocks, games=games)
    resolved['blocks'] = blocks
    plan = resolved.get('blocks_plan')
    if plan is not None:
        resolved['blocks_plan'] = {**plan, 'block_seconds': cfg['block_seconds'], 'block_sizes': cfg['block_sizes'],
                                   'run_stop_seconds': (resolved.get('early_stop') or {}).get('at_seconds', wall)}
    competition['blocks_wall'] = {
        'wall_seconds': round(wall, 3), 'block_seconds': round(cfg['block_seconds'], 3), 'blocks': cfg['blocks'],
        'block_sizes': cfg['block_sizes'], 'roster_games': games,
        'setup_seconds': round(now - float(notebook_start), 1),
        'nominal': {k: nominal.get(k) for k in ('wall_seconds', 'blocks', 'roster_games')},
        'preview_stop_seconds': None if is_rerun() else int(competition['preview_seconds']),
        'rule': 'submission_seconds from the notebook start - measured setup - cleanup_seconds',
    }
    return resolved


def gateway_arcade(url, recordings):
    return Arcade(operation_mode=OperationMode.COMPETITION,
                  arc_base_url=url, arc_api_key='test-key-123',
                  environments_dir='/tmp/arena-no-local-environments',
                  recordings_dir=str(recordings), logger=logging.getLogger('arena.gateway'))


def resolve_opening_cap(resolved, study, roster):
    """Set ``resolved['opening_max_wall_seconds']`` for a gateway roster of ``roster`` games.

    The opening cap of stall-open360 (25 September) is 360 s for 25 games: an equal audition per
    game, so it scales with the roster (110 clones -> 1,584 s). A study with
    ``opening_cap_absolute`` (fixed-2x-gittins-c110, 27 September) states its cap for the roster it
    will play and keeps it unscaled.
    """
    cap = study.get('opening_max_wall_seconds')
    if cap is None:
        return
    if study.get('opening_cap_absolute'):
        resolved['opening_max_wall_seconds'] = float(cap)
        resolved['competition']['opening_cap_scaled'] = {'absolute': True, 'roster': roster, 'seconds': float(cap)}
        return
    base = max(1, len(study['games']))
    resolved['opening_max_wall_seconds'] = float(cap) * roster / base
    resolved['competition']['opening_cap_scaled'] = {
        'per_25_games_seconds': cap, 'roster': roster, 'seconds': resolved['opening_max_wall_seconds']}


@contextmanager
def open_arcade(study, games_dir, output):
    from taaf.competition_arcade import CompetitionArcadeServer

    settings = study['competition']
    os.environ['ONLY_RESET_LEVELS'] = 'true'
    server = None
    owner = None
    try:
        if study.get('quality_eval') and is_rerun():
            raise ValueError('quality_eval is a fixed public research roster, not a competition submission')
        if is_rerun() and not is_rehearsal(study):
            # Connection setup may be retried; environment actions never are.
            deadline = time.monotonic() + 600
            while True:
                try:
                    arcade = gateway_arcade('http://gateway:8001', output / 'recordings')
                    if not arcade.get_environments():
                        raise RuntimeError('Competition gateway returned no games')
                    break
                except Exception:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(2)
        else:
            from .quality_eval import gateway_runs, validate_study

            quality = validate_study(study)
            server = CompetitionArcadeServer(
                game_ids=study['games'], total_runs=gateway_runs(quality) if quality else settings['preview_instances'],
                environments_dir=str(games_dir), save_all_recordings=False,
            ).start()
            arcade = gateway_arcade(server.base_url, output / 'recordings')
        games = [env.game_id for env in arcade.get_environments()]
        if not games or len(set(games)) != len(games):
            raise RuntimeError('Competition roster must have unique nonempty game IDs')
        if (not is_rerun() or is_rehearsal(study)) and len(games) != settings['preview_instances']:
            raise RuntimeError('Public rehearsal roster size mismatch')
        resolved = copy.deepcopy(study)
        resolved['games'] = games
        if study.get('quality_eval'):
            # Version 2 (replicated roster): each exposed clone's source game, replica and seed.
            from .quality_eval import roster
            from .quality_eval import settings as quality_settings

            rows = roster(quality_settings(study['quality_eval']), study['games'], games)
            if rows is not None:
                resolved['quality_roster'] = rows
        resolve_opening_cap(resolved, study, len(games))
        resolved['competition']['execution_mode'] = (
            'rehearsed_hidden_gateway' if is_rehearsal(study) else
            'hidden_gateway' if is_rerun() else 'public_cloned_gateway')
        if not is_rerun() or is_rehearsal(study):
            # Controller-only provenance. Each game worker has its own evidence
            # archive and is never given other replicas' traces or outcomes.
            output.mkdir(parents=True, exist_ok=True)
            (output / 'public-roster.json').write_text(json.dumps([
                {'game_id': game, 'source_game_id': study['games'][i % len(study['games'])]}
                for i, game in enumerate(games)
            ], indent=2))
        owner = ScorecardOwner(arcade, output)
        if server is not None and (settings.get('research_clones')
                                   or (settings.get('preview_live_score') and not is_rerun())):
            owner.local_gateway = server  # live progress reads (ScorecardOwner.peek_score)
        yield resolved, owner
    finally:
        try:
            if owner is not None:
                owner.finalize()
        finally:
            if server is not None:
                server.stop()
