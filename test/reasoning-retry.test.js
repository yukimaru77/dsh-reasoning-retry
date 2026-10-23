import test from 'node:test'
import assert from 'node:assert/strict'
import { Session } from '@deepseek-ai/dsh-session'
import { createSystemMessage, createToolResultMessage, createUserMessage } from '@deepseek-ai/dsh-llm'
import { assertReleasedV4Relationships } from '@deepseek-ai/dsh-session-format-v3-to-v4'
import { apply, clip, failedNodes, renderGuidance, renderRewriteRequest, staleMaxTokens, stepInput } from '../index.js'

const tick = (ms = 20) => new Promise(resolve => setTimeout(resolve, ms))

/** A real DSH Session behind a minimal agent and plugin context. */
function harness({ rewrites = [], goalActivation = 'disarmed', model = 'deepseek-v4' } = {}) {
  const session = Session.create('s1')
  const listeners = new Map()
  const calls = { rewriteRequests: [], resume: [], block: [], notes: [] }
  let turn = 0
  let step = null
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
      if (turn === 1) {
        step = 1
        append('step/start', { turn, step })
        append('system/message', { message: createSystemMessage('SYS'), turn, step }, { surfaceOp: 'append' })
      }
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
    commands: {
      defs: new Map(),
      register(def) { this.defs.set(def.name, def); return () => {} },
      // Mirrors dsh-commands' executor: log-only command/run + command/done around the handler.
      async execute(target, line, attachments, signal) {
        const [, name, rawInput] = line.match(/^\/(\S+)\s?([\s\S]*)$/)
        const def = this.defs.get(name)
        const commandId = `cmd-${calls.commands = (calls.commands ?? 0) + 1}`
        append('command/run', { commandId, name, ...(def.recordInput === false ? {} : { args: rawInput }), source: { kind: 'user' } })
        const result = await def.handler({ commandId, agent: target, rawInput, attachments, signal })
        append('command/done', { commandId, kind: result.kind, text: result.text })
        return { commandId, result }
      },
    },
    systemPrompt: { context: (c) => { calls.context = c; return () => {} } },
    effect: (fn) => fn(),
    on(name, fn) { listeners.set(name, [...(listeners.get(name) ?? []), fn]); return () => {} },
  }
  const api = {
    session, agent, calls, emit, append,
    user(text) { agent.followup(createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'user' } })) },
    assistant(content, at = 1) {
      if (step !== at) {
        if (step !== null) append('step/end', { turn, step })
        step = at
        append('step/start', { turn, step })
      }
      append('assistant/message', { turn, step, message: { id: `a${++assistantId}`, role: 'assistant', content, source: { kind: 'model', provider: 'h200', model } }, stream: [] }, { surfaceOp: 'append' })
      for (const block of content) if (block.type === 'tool-call') append('tool/call', { turn, step, callId: block.id, name: block.name, arguments: block.arguments })
    },
    tool(callId, text) {
      append('tool/result', { turn, step, message: createToolResultMessage({ callId, content: [{ type: 'text', text }], isError: false }) }, { surfaceOp: 'append' })
    },
    end(kind) {
      if (step !== null) append('step/end', { turn, step })
      step = null
      append('turn/end', { turn, reason: { kind } })
    },
    /** The log must pass the stored-format rules, or DSH refuses to reopen the session. */
    assertStorable() {
      const events = session.snapshotEvents()
      assertReleasedV4Relationships({ header: { id: 's1', version: 4 }, events, inheritedEventCount: 0 }, new Set(events.map(e => e.type)))
    },
    async idle() { emit('agent/status', { agent, status: 'idle' }); await tick() },
    cards() { return session.snapshotEvents().filter(e => e.type === 'command/done').map(e => e.data.text) },
    visible() { return session.deriveMessages().map(m => `${m.role}:${m.content.map(b => b.text ?? b.type).join('|')}`) },
  }
  api.ctxCommands = () => ctx.commands
  apply(ctx, { maxRetries: 2 })
  return api
}

test('a cut-off step is removed and re-run with a rewritten prompt; the log keeps the original', async () => {
  const h = harness({ rewrites: ['STEP 1 OF THE TASK'] })
  h.user('BIG TASK')
  h.assistant([{ type: 'reasoning', text: 'circling around option A versus B' }])
  h.end('max-tokens')
  await h.idle()
  assert.deepEqual(h.visible(), ['system:SYS', 'user:[omitted: an earlier attempt that was cut off at the output limit]', 'user:STEP 1 OF THE TASK'])
  h.assertStorable()
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
  assert.deepEqual(h.visible(), ['system:SYS', 'user:[omitted: an earlier attempt that was cut off at the output limit]', 'user:REWRITE 2'])
  const second = h.calls.rewriteRequests[1]
  assert.match(second, /Original input">\nBIG TASK[\s\S]*thinking one[\s\S]*Rewritten input \(attempt 2\)">\nREWRITE 1[\s\S]*thinking two/)
  h.assistant([{ type: 'reasoning', text: 'thinking three' }])
  h.end('max-tokens')
  await h.idle()
  assert.equal(h.calls.block.length, 1)
  assert.equal(h.calls.notes.length, 1)
  assert.equal(h.calls.rewriteRequests.length, 2)
  h.assertStorable()
})

test('a cut-off after tool results keeps the results and adds a rewritten note', async () => {
  const h = harness({ rewrites: ['NOTE: compare the two error lines first'] })
  h.user('DEBUG IT')
  h.assistant([{ type: 'tool-call', id: 'c1', name: 'bash', arguments: '{}' }])
  h.tool('c1', 'error: a\nerror: b')
  h.assistant([{ type: 'reasoning', text: 'reading the log again' }, { type: 'text', text: 'partial' }], 2)
  h.end('max-tokens')
  await h.idle()
  assert.deepEqual(h.visible(), ['system:SYS', 'user:DEBUG IT', 'assistant:tool-call', 'tool:error: a\nerror: b', 'user:[omitted: an earlier attempt that was cut off at the output limit]', 'user:NOTE: compare the two error lines first'])
  h.assertStorable()
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
  assert.match(h.visible()[2], /^user:BIG TASK\n\nWork on this in steps/)
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

test('an automatic re-run leaves an AI-hidden card that the model never sees', async () => {
  const h = harness({ rewrites: ['STEP 1'] })
  h.user('BIG TASK')
  h.assistant([{ type: 'reasoning', text: 'x' }])
  h.end('max-tokens')
  await h.idle()
  const cards = h.cards()
  assert.equal(cards.length, 1)
  assert.match(cards[0], /^🚫 AI非表示：直前の入力と返答（turn 1）/)
  const types = h.session.snapshotEvents().map(e => e.type)
  assert.ok(types.lastIndexOf('command/done') < types.lastIndexOf('user/message'), 'the card sits before the re-run')
  assert.deepEqual(h.visible(), ['system:SYS', 'user:[omitted: an earlier attempt that was cut off at the output limit]', 'user:STEP 1'])
  h.assertStorable()
})

test('/clean-failed-steps hides earlier cut-off replies and old retry notices, keeping the rest', async () => {
  const h = harness()
  h.user('TASK A')
  h.assistant([{ type: 'reasoning', text: 'long' }])
  h.end('max-tokens')
  h.agent.followup(createUserMessage({ content: [{ type: 'text', text: 'OLD NOTICE' }], source: { kind: 'reasoning-retry' } }))
  h.assistant([{ type: 'text', text: 'ok A' }])
  h.end('completed')
  h.user('TASK B')
  h.assistant([{ type: 'tool-call', id: 'c1', name: 'bash', arguments: '{}' }])
  h.tool('c1', 'out')
  h.assistant([{ type: 'reasoning', text: 'long again' }], 2)
  h.end('max-tokens')
  h.user('TASK C')
  h.assistant([{ type: 'text', text: 'ok C' }])
  h.end('completed')
  const { result } = await h.ctxCommands().execute(h.agent, '/clean-failed-steps', [], new AbortController().signal)
  assert.equal(result.kind, 'success')
  assert.match(result.text, /^🚫 AI非表示：出力上限で切れた返答 2件（turn 1〜3）と、古いやり直しの指示 1件/)
  assert.deepEqual(h.visible(), ['system:SYS', 'user:TASK A', 'user:[omitted: an earlier attempt that was cut off at the output limit]', 'assistant:ok A', 'user:TASK B', 'assistant:tool-call', 'tool:out', 'user:[omitted: an earlier attempt that was cut off at the output limit]', 'user:TASK C', 'assistant:ok C'])
  const again = await h.ctxCommands().execute(h.agent, '/clean-failed-steps', [], new AbortController().signal)
  assert.match(again.result.text, /ありません/)
  h.assertStorable()
})

test('failedNodes picks only the last reply of a max-tokens turn', () => {
  const ev = (seq, type, data) => ({ seq, type, data })
  const surface = [ev(1, 'assistant/message', { turn: 1 }), ev(2, 'tool/result', { turn: 1 }), ev(3, 'assistant/message', { turn: 1 }), ev(4, 'assistant/message', { turn: 2 })]
  const { replies } = failedNodes(surface, [...surface, ev(5, 'turn/end', { turn: 1, reason: { kind: 'max-tokens' } }), ev(6, 'turn/end', { turn: 2, reason: { kind: 'completed' } })])
  assert.deepEqual(replies.map(e => e.seq), [3])
})

test('the stored-format rules reject a system/message outside a turn (why placeholders are user messages)', () => {
  const h = harness()
  h.user('TASK')
  h.assistant([{ type: 'text', text: 'done' }])
  h.end('completed')
  h.assertStorable()
  const seq = h.session.surface.nodes.at(-1)
  h.session.append('system/message', { message: createSystemMessage(''), turn: 1, step: 1 }, { surfaceOp: { op: 'replace', startSeq: seq, endSeq: seq }, sourceEventSeqs: [seq] })
  assert.throws(() => h.assertStorable(), /does not match an open turn and step/)
})
