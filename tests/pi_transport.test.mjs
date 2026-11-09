import assert from 'node:assert/strict';
import {Writable} from 'node:stream';
import {createSender, telemetryEvent} from '../pi/transport.mjs';

const partial = {role: 'assistant', content: [{type: 'thinking', thinking: 'x'.repeat(100_000)}]};
const update = {type: 'message_update', message: partial,
  assistantMessageEvent: {type: 'thinking_delta', contentIndex: 0, delta: 'next', partial}};
const original = JSON.stringify(update);
const compact = telemetryEvent(update);
assert.equal(JSON.stringify(update), original); // Pi's own history remains intact.
assert.equal(compact.assistantMessageEvent.delta, 'next');
assert.equal(compact.assistantMessageEvent.contentIndex, 0);
assert.ok(!('partial' in compact.assistantMessageEvent));
assert.ok(!('message' in compact));
assert.ok(JSON.stringify(compact).length < original.length / 100);
const end = {type: 'message_end', message: partial};
assert.equal(telemetryEvent(end), end);
const tool = {type: 'tool_execution_end', result: {content: [{type: 'text', text: 'receipt'}]}};
assert.equal(telemetryEvent(tool), tool);

// A deliberately slow consumer: output order and RPC packets must survive, and
// Node's internal socket queue must never collect an unbounded writev batch.
const received = [];
let outstanding = 0, peak = 0;
const sink = new Writable({highWaterMark: 16, write(chunk, encoding, callback) {
  received.push(JSON.parse(chunk));
  setTimeout(callback, 1);
}});
const rawWrite = sink.write.bind(sink);
sink.write = (chunk, callback) => {
  peak = Math.max(peak, ++outstanding);
  return rawWrite(chunk, error => { --outstanding; callback(error); });
};
const send = createSender(sink);
const promises = [];
for (let i = 0; i < 500; ++i) promises.push(send({type: 'event', i, event: compact}));
const receipt = {type: 'rpc', id: 77, operation: 'metadata'};
promises.push(send(receipt));
receipt.id = -1; // Queue stores the submitted packet, not a mutable SDK reference.
promises.push(send({type: 'event', event: end}));
promises.push(send({type: 'done'}));
await Promise.all(promises);
assert.equal(peak, 1);
assert.deepEqual(received.slice(0, 500).map(p => p.i), Array.from({length: 500}, (_, i) => i));
assert.equal(received[500].id, 77);
assert.deepEqual(received[501].event, end);
assert.equal(received[502].type, 'done');

const broken = new Writable({write(chunk, encoding, callback) {
  callback(Object.assign(new Error('closed'), {code: 'EPIPE'}));
}});
const failedSend = createSender(broken);
await assert.rejects(failedSend({type: 'rpc'}), {code: 'EPIPE'});
await assert.rejects(failedSend({type: 'done'}), {code: 'EPIPE'});
console.log(JSON.stringify({tests: 'passed', packets: received.length,
  peakOutstandingWrites: peak, originalUpdateBytes: original.length,
  compactUpdateBytes: JSON.stringify(compact).length, node: process.version}));
