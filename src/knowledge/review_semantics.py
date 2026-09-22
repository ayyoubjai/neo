"""Apply explicit, auditable interpretation corrections; never endorse propositions."""
import argparse
import json
from pathlib import Path
from knowledge.documents import stable_id

ALLOWED = {'text', 'scope', 'stance', 'claim_type', 'concepts', 'entities', 'entity_kinds', 'semantic_flags'}


def validate_review(payload):
    if not isinstance(payload.get('reviewer'), str) or not payload['reviewer'].strip():
        raise ValueError('Identify the actual reviewer')
    for change in payload.get('claims', []):
        if not change.get('note') or not change.get('claim_id') or not isinstance(change.get('correction'), dict):
            raise ValueError('Claim corrections require target, fields and justification')
        correction=change['correction']
        if set(correction)-ALLOWED:
            raise ValueError('Cannot alter claim identity, source quotations or evidence through a semantic review')
        for field in ('text','scope','stance','claim_type'):
            if field in correction and (not isinstance(correction[field],str) or not correction[field].strip()):
                raise ValueError('Correction fields must be nonempty strings')
        for field in ('concepts','entities','entity_kinds','semantic_flags'):
            if field in correction and (not isinstance(correction[field],list) or any(not isinstance(x,str) for x in correction[field])):
                raise ValueError('Term fields must be lists of strings')
        if ('entities' in correction) != ('entity_kinds' in correction):
            raise ValueError('Entity names and kinds must be corrected together')
        from knowledge.semantics import ENTITY_KINDS
        if len(correction.get('entities',[])) != len(correction.get('entity_kinds',[])) or any(k not in ENTITY_KINDS for k in correction.get('entity_kinds',[])):
            raise ValueError('Entity names and supported kinds must align')
        from knowledge.extraction import STANCES, CLAIM_TYPES
        if 'stance' in correction and correction['stance'] not in STANCES or 'claim_type' in correction and correction['claim_type'] not in CLAIM_TYPES:
            raise ValueError('Unknown stance or claim type')
    for decision in payload.get('relationships', []):
        if not decision.get('id') or not decision.get('note') or decision.get('status') not in {'retained','rejected'}:
            raise ValueError('Relationship reviews require ID, status and reason')
    from knowledge.extraction import RELATIONS
    for edge in payload.get('new_relationships', []):
        if not edge.get('source_id') or not edge.get('target_id') or edge.get('relation') not in RELATIONS or not edge.get('note'):
            raise ValueError('A replacement link requires endpoints, a supported relation and justification')
    for group in payload.get('propositions', []):
        if not group.get('text') or not group.get('scope') or not group.get('note') or len(set(group.get('claim_ids',[])))<2:
            raise ValueError('Grouping requires explicit proposition, scope, justification and at least two source claims')
    return stable_id('semantic-review', json.dumps(payload, sort_keys=True))


def apply_review(graph, payload):
    review_id=validate_review(payload)
    import time
    def persist(model):
        with model.driver.session() as tx:
            if tx.run('MATCH (r:SemanticReview {id:$id}) RETURN r.id',id=review_id).single():
                return
            # Validate every reference before making a partial review visible.
            ids=set(c['claim_id'] for c in payload.get('claims',[]))
            ids.update(i for g in payload.get('propositions',[]) for i in g['claim_ids'])
            ids.update(i for e in payload.get('new_relationships',[]) for i in (e['source_id'],e['target_id']))
            found=tx.run('MATCH (c:Claim) WHERE c.id IN $ids RETURN collect(c.id) AS ids',ids=sorted(ids)).single()['ids']
            if set(found)!=ids:raise ValueError('Unknown claim in semantic review')
            for d in payload.get('relationships',[]):
                if tx.run('MATCH ()-[r:CLAIM_RELATION {id:$id}]->() RETURN r.id',id=d['id']).single() is None:
                    raise ValueError('Unknown relationship in semantic review')
            tx.run('MERGE (r:SemanticReview {id:$id}) ON CREATE SET r.payload_json=$payload,r.reviewer=$reviewer,r.created_at=$now',id=review_id,payload=json.dumps(payload),reviewer=payload['reviewer'],now=time.time()).consume()
            for change in payload.get('claims',[]):
                tx.run('MATCH (r:SemanticReview {id:$review}), (c:Claim {id:$id}) MERGE (r)-[e:INTERPRETS]->(c) SET e.correction_json=$correction,e.note=$note,e.context_json=$context,c.review_required=true',review=review_id,id=change['claim_id'],correction=json.dumps(change['correction']),note=change['note'],context=json.dumps(change.get('supporting_context',{}))).consume()
                correction=change['correction']
                if 'concepts' in correction:
                    tx.run('MATCH (c:Claim {id:$id}) SET c.search_concepts=$concepts WITH c UNWIND $concepts AS name MERGE (t:Concept {name:name}) MERGE (c)-[:REVIEWED_ABOUT {review_id:$review}]->(t)',id=change['claim_id'],concepts=correction['concepts'],review=review_id).consume()
                if 'text' in correction:
                    tx.run('MATCH (c:Claim {id:$id}) SET c.search_text=$text',id=change['claim_id'],text=correction['text']).consume()
                for name,kind in zip(correction.get('entities',[]),correction.get('entity_kinds',[])):
                    tx.run('MATCH (c:Claim {id:$id}) MERGE (e:Entity {id:$entity}) ON CREATE SET e.name=$name,e.kind=$kind MERGE (c)-[:REVIEWED_MENTIONS {review_id:$review}]->(e)',id=change['claim_id'],entity=stable_id('entity',kind,name),name=name,kind=kind,review=review_id).consume()
                tx.run("MATCH (i:Investigation {status:'approved'})-[:INVESTIGATES]->(:Claim {id:$id}) SET i.status='planned' REMOVE i.approval_hash",id=change['claim_id']).consume()
            for decision in payload.get('relationships',[]):
                tx.run('MATCH ()-[e:CLAIM_RELATION {id:$id}]->() SET e.review_status=$status,e.review_note=$note,e.semantic_review_id=$review',id=decision['id'],status=decision['status'],note=decision['note'],review=review_id).consume()
            for edge in payload.get('new_relationships',[]):
                eid=stable_id('reviewed-link',review_id,edge['source_id'],edge['target_id'],edge['relation'])
                tx.run("MATCH (a:Claim {id:$a}),(b:Claim {id:$b}) MERGE (a)-[e:CLAIM_RELATION {id:$id}]->(b) ON CREATE SET e.kind=$kind,e.reasoning=$note,e.scope_match='reviewed_context',e.status='proposed',e.origin='semantic_review',e.review_status='retained',e.semantic_review_id=$review",a=edge['source_id'],b=edge['target_id'],id=eid,kind=edge['relation'],note=edge['note'],review=review_id).consume()
            for group in payload.get('propositions',[]):
                pid=stable_id('proposition',group['text'],group['scope'])
                tx.run('MERGE (p:Proposition {id:$id}) ON CREATE SET p.text=$text,p.scope=$scope WITH p MATCH (r:SemanticReview {id:$review}) MERGE (r)-[:PROPOSES_GROUP]->(p) WITH p UNWIND $claims AS id MATCH (c:Claim {id:id}) MERGE (c)-[e:EXPRESSES_PROPOSITION]->(p) SET e.review_id=$review,e.note=$note,e.status="reviewed_interpretation"',id=pid,text=group['text'],scope=group['scope'],review=review_id,claims=group['claim_ids'],note=group['note']).consume()
    graph.run_transaction(persist)
    return review_id


def main():
    from knowledge.graph import KnowledgeGraph
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('review',type=Path);args=p.parse_args()
    graph=KnowledgeGraph()
    try:print(apply_review(graph,json.loads(args.review.read_text())))
    finally:graph.close()

if __name__=='__main__':main()
