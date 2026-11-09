// Actual pinned Pi SDK. Tool definitions and coding loop remain Pi-owned.
import readline from 'node:readline';
import path from 'node:path';
import {createSender, telemetryEvent} from '../pi/transport.mjs';
import {
  createAgentSession, ModelRuntime, SettingsManager, SessionManager, DefaultResourceLoader,
  createReadToolDefinition, createWriteToolDefinition, createEditToolDefinition,
  createBashToolDefinition,
} from '@earendil-works/pi-coding-agent';
// The SDK sets provider timeouts but, unlike Pi's CLI, does not configure HTTP.
// This internal module is pinned with Pi 0.85.1 by the dependency lockfile.
const {configureHttpDispatcher} = await import(new URL('core/http-dispatcher.js',
  import.meta.resolve('@earendil-works/pi-coding-agent')));

if (process.version !== 'v22.19.0') throw new Error('Pinned Node 22.19.0 required');
const writePacket = createSender(process.stdout);
const send = value => writePacket(value).catch(error => {
  process.stderr.write(`Pi transport failed: ${error.stack || error}\n`);
  process.exit(1); // Host retains acknowledged receipts; never replay a failed turn.
});
let sequence = 0;
const pending = new Map();
function rpc(operation, args = {}) {
  const id = ++sequence;
  return new Promise((resolve, reject) => {
    pending.set(id, {resolve, reject});
    send({type: 'rpc', id, operation, ...args});
  });
}
let session;
async function initialize(cfg) {
  const cwd = cfg.workspace;
  const settingsManager = SettingsManager.inMemory({
    compaction: {enabled: true, reserveTokens: 16384, keepRecentTokens: 20000},
    retry: {enabled: false},
    httpIdleTimeoutMs: 0,
  });
  configureHttpDispatcher(settingsManager.getHttpIdleTimeoutMs());
  const resourceLoader = new DefaultResourceLoader({cwd, agentDir: cfg.agentDir,
    settingsManager, noExtensions: true, noSkills: true, noPromptTemplates: true,
    noThemes: true, noContextFiles: true, appendSystemPrompt: [cfg.system],
  });
  await resourceLoader.reload();
  const runtime = await ModelRuntime.create({authPath: path.join(cfg.agentDir, 'auth.json'),
    modelsPath: null, allowModelNetwork: false, refreshOnCreate: false});
  runtime.registerProvider('arc-local', {api: 'openai-completions', baseUrl: cfg.baseUrl,
    apiKey: 'local-only', models: [{id: cfg.model, name: 'Pinned local Qwen', reasoning: true,
      input: cfg.textOnly ? ['text'] : ['text', 'image'], cost: {input: 0, output: 0, cacheRead: 0, cacheWrite: 0},
      contextWindow: cfg.contextWindow ?? 262144, maxTokens: cfg.contextWindow ?? 262144,
      compat: {supportsDeveloperRole: false, supportsStore: false,
        thinkingFormat: 'qwen-chat-template', supportsReasoningEffort: false,
        supportsThinkingTokenBudget: false},
    }]});
  const fileOps = {
    readFile: async p => Buffer.from((await rpc(
      cfg.textOnly && p.toLowerCase().endsWith('.png') ? 'read_grid' : 'read',
      {path: p})).output.trim(), 'base64'),
    writeFile: async (p, content) => { await rpc('write', {path: p, content}); },
    mkdir: async p => { await rpc('mkdir', {path: p}); },
    access: async p => { await rpc('access', {path: p}); },
    detectImageMimeType: async p => !cfg.textOnly && p.toLowerCase().endsWith('.png') ? 'image/png' : null,
  };
  const definitions = [
    createReadToolDefinition(cwd, {operations: fileOps, autoResizeImages: false}),
    createWriteToolDefinition(cwd, {operations: fileOps}),
    createEditToolDefinition(cwd, {operations: fileOps}),
    createBashToolDefinition(cwd, {exposeSessionEnvironment: false, operations: {
      exec: async (command, cwd, options) => {
        // Pi passes milliseconds; the executor clips seconds by its configured maximum.
        const result = await rpc('bash', {command,
          timeout: options.timeout ? options.timeout / 1000 : undefined});
        options.onData(Buffer.from(result.output));
        return {exitCode: result.exitCode};
      },
    }}),
  ].map(def => ({...def, execute: async (...args) => {
    const before = await rpc('metadata');
    try {
      const result = await def.execute(...args);
      const after = await rpc('metadata', {since: before.index});
      if (after.observation_id !== before.observation_id)
        result.content.push({type: 'text', text: 'Acknowledged execution evidence: ' + JSON.stringify(after)});
      return result;
    } catch (error) {
      const after = await rpc('metadata', {since: before.index});
      throw new Error(String(error) + '\nCurrent authoritative execution evidence: ' + JSON.stringify(after));
    }
  }}));
  // pi-rlm: the host-owned persistent Python kernel (``env`` steps through the game bridge).
  // Not wrapped in the evidence acknowledgment above; registered only when the host lists
  // 'python' in cfg.tools, so every other kind's session is unchanged.
  definitions.push({
    name: 'python', label: 'python',
    description: 'Run Python in this game\'s persistent kernel: variables, functions and imports '
      + 'persist across calls; env (the game), np (numpy) and skills (the shared harness '
      + 'procedures) are predefined. Returns what the code prints, truncated at 4,000 '
      + 'characters: keep large results in variables and print selectively.',
    parameters: {type: 'object', properties: {
      code: {type: 'string', description: 'Python source to execute in the persistent kernel'},
      timeout: {type: 'number', description: 'Seconds allowed for this call (optional; the host clips it)'},
    }, required: ['code'], additionalProperties: false},
    execute: async (toolCallId, params) => {
      const result = await rpc('python', {code: params.code, timeout: params.timeout});
      return {content: [{type: 'text', text: result.output}], details: result.details};
    },
  });
  const enabled = cfg.tools || ['read', 'write', 'edit', 'bash'];
  ({session} = await createAgentSession({cwd, agentDir: cfg.agentDir, modelRuntime: runtime,
    model: runtime.getModel('arc-local', cfg.model), thinkingLevel: 'high',
    resourceLoader, settingsManager, sessionManager: SessionManager.create(cwd, cfg.sessionDir),
    tools: enabled, customTools: definitions.filter(def => enabled.includes(def.name))}));
  session.subscribe(event => send({type: 'event', event: telemetryEvent(event)}));
  send({type: 'ready'});
}

const lines = readline.createInterface({input: process.stdin});
for await (const line of lines) {
  const msg = JSON.parse(line);
  if (msg.type === 'reply') {
    const promise = pending.get(msg.id);
    pending.delete(msg.id);
    if (msg.error) promise.reject(new Error(msg.error)); else promise.resolve(msg.result);
  } else if (msg.type === 'init') {
    initialize(msg).catch(error => send({type: 'fatal', error: String(error), stack: error.stack}));
  } else if (msg.type === 'prompt') {
    session.prompt(msg.text, {images: msg.images}).then(() => send({type: 'done'}))
      .catch(error => send({type: 'fatal', error: String(error), stack: error.stack}));
  } else if (msg.type === 'interrupt') {
    // Stop the running prompt (turn cap or deadline); the session survives for resumption.
    session.abort().then(() => send({type: 'interrupted'}))
      .catch(error => send({type: 'fatal', error: String(error)}));
  } else if (msg.type === 'compact') {
    session.compact().then(result => send({type: 'compacted', result}))
      .catch(error => send({type: 'fatal', error: String(error)}));
  } else if (msg.type === 'abort') {
    await session.abort(); session.dispose(); break;
  }
}
