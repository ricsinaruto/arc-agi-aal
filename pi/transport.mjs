// Transport telemetry only; Pi's session and model history are never modified.
export function telemetryEvent(event) {
  if (event.type !== 'message_update') return event;
  const {message, assistantMessageEvent, ...rest} = event;
  const {partial, ...delta} = assistantMessageEvent || {};
  return {...rest, assistantMessageEvent: delta,
    omittedSnapshots: 'message and partial; deltas retained, full message at message_end'};
}

export function createSender(stream) {
  let tail = Promise.resolve();
  let failure;
  stream.on('error', error => { failure = error; });
  return value => {
    // Serialize now: SDK event objects may be mutated by the next streamed chunk.
    const line = JSON.stringify(value) + '\n';
    const result = tail.then(() => {
      if (failure) throw failure;
      // One outstanding write, even when the receiver is slow. Do not accumulate
      // an enormous writev batch inside Node's stdout socket.
      return new Promise((resolve, reject) => {
        stream.write(line, error => error ? reject(error) : resolve());
      });
    });
    tail = result.catch(error => { failure = error; });
    return result;
  };
}
