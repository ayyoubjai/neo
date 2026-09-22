"""Bounded background preparation and execution of reviewed claim investigations."""
import argparse
import asyncio
from dataclasses import dataclass
import json
import logging
import time
import uuid

from autonomy.cognition_client import CognitionClient
from knowledge.graph import KnowledgeGraph
from knowledge.investigate import investigate, resume, default_journal


@dataclass
class WorkerConfig:
    collection: str
    pause_s: float = 60
    max_plans_per_day: int = 4
    max_investigations_per_day: int = 4
    max_tool_calls_per_day: int = 12
    max_actions_per_investigation: int = 3
    reassess_after_days: float = 30
    objective: str = ''

    def __post_init__(self):
        import math
        if not isinstance(self.collection, str) or not self.collection.strip():
            raise ValueError('Knowledge worker requires a collection')
        for name in ('max_plans_per_day', 'max_investigations_per_day', 'max_tool_calls_per_day', 'max_actions_per_investigation'):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if not 1 <= self.max_actions_per_investigation <= 10:
            raise ValueError('Action limit must be 1..10')
        if not math.isfinite(self.pause_s) or self.pause_s < 1 or not math.isfinite(self.reassess_after_days) or self.reassess_after_days < 0:
            raise ValueError('Pause must be >=1 second and reassessment age nonnegative')


async def tick(graph, client, config, journal):
    result = {'planned': [], 'assessed': [], 'issues': []}
    # Shared journal lock prevents concurrent ticks exceeding preparation budgets.
    with journal.lock('worker:' + config.collection):
        pending = await asyncio.to_thread(graph.pending_plans, config.collection, limit=10000)
        for item in pending:
            if item['status'] == 'planned':
                continue
            try:
                saved = await asyncio.to_thread(graph.get_plan, item['id'])
                actions = saved['investigation'].get('max_actions', 3)
                if item['status'] == 'approved':
                    if actions > config.max_actions_per_investigation:
                        result['issues'].append({'id': item['id'], 'reason': 'Approved action allowance exceeds worker limit'})
                        continue
                    if not journal.reserve('execute:' + config.collection, item['id'],
                        max_investigations=config.max_investigations_per_day,
                        max_actions=config.max_tool_calls_per_day, actions=actions):
                        continue
                # Executing plans may only replay durable receipts, never uncertain tool calls.
                completed = await resume(graph, client, item['id'], journal=journal)
                result['assessed'].append(completed['assessment']['id'])
            except Exception as exc:
                result['issues'].append({'id': item['id'], 'reason': str(exc)})
        active = {p['claim_id'] for p in pending}
        queue = await asyncio.to_thread(graph.queue, config.collection, limit=1000)
        terms = set(config.objective.casefold().split())
        queue.sort(key=lambda c: (c.get('last_assessed') is not None,
            -c.get('conflicts', 0), -len(terms & set(c['text'].casefold().split())), c.get('last_assessed') or 0))
        cutoff = time.time() - config.reassess_after_days * 86400
        for claim in queue:
            if (claim['id'] in active or claim['claim_type'] != 'empirical' or claim.get('review_required')
                    or claim.get('last_assessed') is not None and claim['last_assessed'] > cutoff):
                continue
            # Charge failed planning attempts too, so errors cannot create an unbounded model loop.
            if not journal.reserve('plan:' + config.collection, uuid.uuid4().hex,
                max_investigations=config.max_plans_per_day, max_actions=0, actions=0):
                break
            try:
                planned = await investigate(graph, client, claim['id'], max_actions=config.max_actions_per_investigation)
                result['planned'].append(planned['investigation']['id'])
            except Exception as exc:
                result['issues'].append({'id': claim['id'], 'reason': str(exc)})
    return result


async def run_worker(settings, *, once=False, config=None):
    config = config or WorkerConfig(**dict(settings.autonomy.get('knowledge_worker') or {}))
    client = CognitionClient(settings=settings, permission_policy='deny')
    await client.wait_until_available()
    graph, journal = KnowledgeGraph(), default_journal(settings)
    try:
        while True:
            try:
                result = await tick(graph, client, config, journal)
                print('[knowledge-worker] ' + json.dumps(result))
            except Exception:
                if once:
                    raise
                logging.exception('Knowledge worker tick failed')
            if once:
                return result
            await asyncio.sleep(config.pause_s)
    finally:
        graph.close()


def main():
    from common.config import load_settings
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--collection')
    args = parser.parse_args()
    settings = load_settings()
    values = dict(settings.autonomy.get('knowledge_worker') or {})
    if args.collection:
        values['collection'] = args.collection
    asyncio.run(run_worker(settings, once=args.once, config=WorkerConfig(**values)))


if __name__ == '__main__':
    main()
