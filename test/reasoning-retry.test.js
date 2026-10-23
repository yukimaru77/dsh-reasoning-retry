import test from 'node:test'
import assert from 'node:assert/strict'
import { Session } from '@deepseek-ai/dsh-session'
import { createSystemMessage, createToolResultMessage, createUserMessage } from '@deepseek-ai/dsh-llm'
import { apply, clip, renderGuidance, renderRewriteRequest, staleMaxTokens, stepInput } from '../index.js'

const tick = (ms = 20) => new Promise(resolve => setTimeout(resolve, ms))

/** A real DSH Session behind a minimal agent and plugin context. */
function harness({ rewrites = [], goalActivation = 'disarmed', model = 'deepseek-v4' } = {}) {
  const session = Session.create('s1')
  const listeners = new Map()
  const calls = { rewriteRequests: [], resume: [], block: [], notes: [] }
  let turn = 0
  let assistantId = 0
  const emit = (name, ...args) => { for (const fn of listeners.get(name) ?? []) fn(...args) }
  const append = (type, data, opts) => {
    session.append(type, data, opts)
    emit('session/event', session, session.snapshotEvents().at(-1))
  }
  const agent = {
    id: 's1',
    session,
    status: 'idle',
    options: { provider: 'h200', model },
    followup(message) {
      if (message.source.kind === 'reasoning-retry-note') { calls.notes.push(message); return }
      turn++
      append('turn/start', { turn })
      append('user/message', message, { surfaceOp: 'append' })
    },
    inject(message) { calls.notes.push(message) },
  }
  session.requestHeader = () => ({ config: { provider: 'h200', model, maxTokens: 32768 } })
  const goal = { id: 'g1', revision: 1, phase: 'active', activation: goalActivation, objective: 'G' }
  const ctx = {
    logger: { info() {}, warn() {} },
    agents: { get: id => id === 's1' ? agent : undefined },
    goals: { get: () => goal, resume: (...a) => calls.resume.push(a), block: (...a) => calls.block.push(a) },
    llm: {
      stream(options) {
        calls.rewriteRequests.push(options.messages[0].content[0].text)
        const text = rewrites.shift()
        return (async function* () { if (text !== undefined) yield { type: 'text-delta', index: 0, text } })()
      },
    },
    systemPrompt: { context: (c) => { calls.context = c; return () => {} } },
    effect: (fn) => fn(),
    on(name, fn) { listeners.set(name, [...(listeners.get(name) ?? []), fn]); return () => {} },
  }
  append('system/message', { message: createSystemMessage('SYS'), turn: 0, step: 1 }, { surfaceOp: 'append' })
  const api = {
    session, agent, calls, emit, append,
    user(text) { agent.followup(createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'user' } })) },
    assistant(content, step = 1) {
      append('assistant/message', { turn, step, message: { id: `a${++assistantId}`, role: 'assistant', content, source: { kind: 'model', provider: 'h200', model } }, stream: [] }, { surfaceOp: 'append' })
    },
    tool(callId, text, step = 1) {
      append('tool/result', { turn, step, message: createToolResultMessage({ callId, content: [{ type: 'text', text }], isError: false }) }, { surfaceOp: 'append' })
    },
    end(kind) { append('turn/end', { turn, reason: { kind } }) },
    async idle() { emit('agent/status', { agent, status: 'idle' }); await tick() },
    visible() { return session.deriveMessages().map(m => `${m.role}:${m.content.map(b => b.text ?? b.type).join('|')}`) },
  }
  apply(ctx, { maxRetries: 2 })
  return api
}

test('a cut-off step is removed and re-run with a rewritten prompt; the log keeps the original', async () => {
  const h = harness({ rewrites: ['STEP 1 OF THE TASK'] })
  h.user('BIG TASK')
  h.assistant([{ type: 'reasoning', text: 'circling around option A versus B' }])
  h.end('max-tokens')
  await h.idle()
  assert.deepEqual(h.visible(), ['system:SYS', 'user:STEP 1 OF THE TASK'])
  const request = h.calls.rewriteRequests[0]
  assert.match(request, /<input label="Original input">\nBIG TASK/)
  assert.match(request, /circling around option A versus B/)
  assert.match(request, /Do not tell the agent to think less/)
  assert.ok(h.session.snapshotEvents().some(e => e.type === 'user/message' && e.data.content[0].text === 'BIG TASK'), 'the log keeps it')
})

test('a second cut-off is rewritten from the whole chain, then the ladder gives up', async () => {
  const h = harness({ rewrites: ['REWRITE 1', 'REWRITE 2'] })
  h.user('BIG TASK')
  h.assistant([{ type: 'reasoning', text: 'thinking one' }])
  h.end('max-tokens')
  await h.idle()
  h.assistant([{ type: 'reasoning', text: 'thinking two' }])
  h.end('max-tokens')
  await h.idle()
  assert.deepEqual(h.visible(), ['system:SYS', 'user:REWRITE 2'])
  const second = h.calls.rewriteRequests[1]
  assert.match(second, /Original input">\nBIG TASK[\s\S]*thinking one[\s\S]*Rewritten input \(attempt 2\)">\nREWRITE 1[\s\S]*thinking two/)
  h.assistant([{ type: 'reasoning', text: 'thinking three' }])
  h.end('max-tokens')
  await h.idle()
  assert.equal(h.calls.block.length, 1)
  assert.equal(h.calls.notes.length, 1)
  assert.equal(h.calls.rewriteRequests.length, 2)
})

test('a cut-off after tool results keeps the results and adds a rewritten note', async () => {
  const h = harness({ rewrites: ['NOTE: compare the two error lines first'] })
  h.user('DEBUG IT')
  h.assistant([{ type: 'tool-call', id: 'c1', name: 'bash', arguments: { command: 'make' } }])
  h.tool('c1', 'error: a\nerror: b', 1)
  h.assistant([{ type: 'reasoning', text: 'reading the log again' }, { type: 'text', text: 'partial' }], 2)
  h.end('max-tokens')
  await h.idle()
  assert.deepEqual(h.visible(), ['system:SYS', 'user:DEBUG IT', 'assistant:tool-call', 'tool:error: a\nerror: b', 'user:NOTE: compare the two error lines first'])
  assert.match(h.calls.rewriteRequests[0], /<tool-output label="Original input">\nerror: a/)
  assert.match(h.calls.rewriteRequests[0], /<partial-visible-output>\npartial/)
})

test('a successful re-run ends the ladder and resumes the goal', async () => {
  const h = harness({ rewrites: ['REWRITE 1'] })
  h.user('BIG TASK')
  h.assistant([{ type: 'reasoning', text: 'x' }])
  h.end('max-tokens')
  await h.idle()
  h.assistant([{ type: 'text', text: 'done' }])
  h.end('completed')
  assert.equal(h.calls.resume.length, 1)
  h.user('NEXT')
  h.assistant([{ type: 'reasoning', text: 'y' }])
  h.end('max-tokens')
  await h.idle()
  assert.match(h.calls.rewriteRequests[1], /Original input">\nNEXT/)
})

test('without a rewrite the step re-runs with a deterministic step-by-step note', async () => {
  const h = harness({ rewrites: [] })
  h.user('BIG TASK')
  h.assistant([{ type: 'reasoning', text: 'x' }])
  h.end('max-tokens')
  await h.idle()
  assert.match(h.visible()[1], /^user:BIG TASK\n\nWork on this in steps/)
})

test('stepInput skips injected context and finds tool inputs', () => {
  const surface = [
    { seq: 1, type: 'assistant/message', data: {} },
    { seq: 2, type: 'user/message', data: { source: { kind: 'goal' }, content: [{ type: 'text', text: 'goal prompt' }] } },
    { seq: 3, type: 'user/message', data: { source: { kind: 'time-context' }, content: [{ type: 'text', text: 'time' }] } },
    { seq: 4, type: 'assistant/message', data: {} },
  ]
  assert.deepEqual(stepInput(surface, 4), { kind: 'prompt', text: 'goal prompt', deleteFrom: 2 })
  assert.equal(clip('x'.repeat(100), 20).includes('characters omitted'), true)
  assert.match(renderRewriteRequest('prompt', [{ input: 'a', reasoning: 'b' }]), /Output only the text of the new message/)
})

test('a cap once forced below the adapter default is dropped', () => {
  const headers = [{ config: { maxTokens: 32768 }, adapterDefaults: { maxTokens: true } }, { config: { maxTokens: 4096 } }]
  assert.equal(staleMaxTokens(headers, 4096), 32768)
  assert.equal(staleMaxTokens(headers, 32768), undefined)
  assert.equal(staleMaxTokens([{ config: { maxTokens: 4096 } }], 4096), undefined)
})

test('standing guidance names the budget and applies to DeepSeek only', () => {
  const h = harness()
  assert.match(h.calls.context.text({ scope: h.agent }), /output budget per reply is 32768 tokens/)
  const codex = harness({ model: 'gpt-6.1-sol' })
  assert.equal(codex.calls.context.text({ scope: codex.agent }), '')
  assert.match(renderGuidance(undefined), /limited/)
})
