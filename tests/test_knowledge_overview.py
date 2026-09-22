"""Graph projection checks independent of browser layout or a Neo4j service."""
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('node'), 'requires Node.js')
class OverviewTests(unittest.TestCase):
    def test_disconnected_nodes_direction_deduplication_and_optional_layers(self):
        script = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const context={};vm.createContext(context);
vm.runInContext(fs.readFileSync('src/knowledge/explorer-graph.js','utf8'),context);
const claim=(id)=>({claim:{id,text:id,concepts:['shared']},sources:[{document:'Book',document_hash:'hash'}],relationships:[],assessments:[]});
const a=claim('a'),b=claim('b'),isolated=claim('isolated');
const relationship={source_id:'a',target_id:'b',relationship:{id:'ab',kind:'SUPPORTS'}};
a.relationships=[relationship];b.relationships=[relationship];
a.assessments=[{assessment:{id:'assessment',position:'supported',reasoning:'test'}}];
const records=[a,b,isolated];
let graph=context.layoutOverview(context.buildOverview(records));
assert.equal(graph.nodes.length,3);assert.equal(graph.edges.length,1);assert.equal(graph.components,2);
assert.equal(graph.byId.get(graph.edges[0].a).label,'a');
assert.equal(graph.byId.get(graph.edges[0].b).label,'b');
assert(graph.nodes.every(n=>Number.isFinite(n.x)&&Number.isFinite(n.y)));
graph=context.layoutOverview(context.buildOverview(records,{concepts:true,sources:true,assessments:true}));
assert.equal(graph.nodes.length,6);assert.equal(graph.edges.length,8);assert.equal(graph.components,1);
assert.equal(graph.nodes.find(n=>n.kind==='source').claimIds.length,3);
// A shared title does not collapse documents with different hashes.
b.sources=[{document:'Book',document_hash:'other-hash'}];
assert.equal(context.buildOverview(records,{sources:true}).nodes.filter(n=>n.kind==='source').length,2);
a.relationships.push({source_id:'a',target_id:'outside',relationship:{id:'outside',kind:'RELATED_TO'}});
graph=context.layoutOverview(context.buildOverview(records));
assert.equal(graph.omitted,1);assert.equal(graph.edges.length,1);
assert.equal(context.layoutOverview(context.buildOverview([])).components,0);
a.propositions=[{id:'common',text:'Reviewed proposition'}];b.propositions=a.propositions;
a.claim.entities=['Darwin'];a.claim.entity_kinds=['person'];
graph=context.buildOverview(records,{entities:true,propositions:true});
assert.equal(graph.nodes.filter(n=>n.kind==='proposition').length,1);
assert.equal(graph.nodes.find(n=>n.kind==='proposition').claimIds.length,2);
assert.equal(graph.nodes.filter(n=>n.kind==='entity').length,1);
graph=context.layoutOverview(context.buildOverview(records,{concepts:true,connectionTypes:['ABOUT']}));
assert.equal(graph.edges.length,3);assert(graph.edges.every(e=>e.kind==='ABOUT'));
graph=context.layoutOverview(context.buildOverview(records,{concepts:true,connectionTypes:[]}));
assert.equal(graph.edges.length,0);assert.equal(graph.nodes.length,4);assert.equal(graph.components,4);
assert.notEqual(context.connectionColor('SUPPORTS'),context.connectionColor('POTENTIALLY_CONTRADICTS'));
'''
        subprocess.run(['node', '-e', script], cwd=ROOT, check=True, capture_output=True, text=True)
