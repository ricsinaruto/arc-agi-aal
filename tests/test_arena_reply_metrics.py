from arc_infra.fleet import fleet_summary


def test_fleet_summary_counts_completed_replies_and_their_tokens():
    row = dict(game_id='game', score=1, actions=5, levels=1, status='time_limit', error=None,
               model_calls=3, completed_replies=2, completion_tokens=100)
    report = fleet_summary([row], 1800)
    assert report['completed_replies'] == 2 and report['completed_reply_tokens'] == 100
    assert report['mean_score'] == 1 and report['actions'] == 5
    # Duck and historical adapters retain their existing completed-call semantics.
    row.pop('completed_replies')
    row['model_calls'] = 2
    assert fleet_summary([row], 1800)['completed_replies'] == 2
