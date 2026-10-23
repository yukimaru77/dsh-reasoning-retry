import test from 'node:test'
import assert from 'node:assert/strict'
import {
  apply,
  isReasoningOnly,
  repetitionStats,
  renderGuidance,
  renderRetryMessage,
  staleMaxTokens,
} from '../index.js'

const tick = () => new Promise(resolve => setTimeout(resolve, 20))

function makeHarness(options = {}) {
  const listeners = new Map()
  const calls = { resume: [], block: [], followups: [], injected: [], contexts: [] }
  const goal = {
    id: 'goal-test',
    revision: 1,
    phase: options.goalPhase ?? 'active',
    activation: options.goalActivation ?? 'disarmed',
    objective: 'Test goal',
  }
  const session = {
    id: 'session-test',
    requestHeader: () => ({ config: { provider: options.provider ?? 'octane-preview', model: options.model ?? 'deepseek/deepseek-v4-1-flash-1017', maxTokens: 32768 } }),
  }
  const agent = {
    id: 'agent-test',
    session,
    status: 'idle',
    options: { provider: 'p', model: 'm' },
    inbox: { nextTurn: [], nextStep: [] },
    followup(message) { calls.followups.push(message) },
    inject(message) { calls.injected.push(message) },
  }
  const ctx = {
    logger: { info() {}, warn() {} },
    agents: { get: id => id === session.id ? agent : undefined, list: () => [agent] },
    goals: {
      get: () => goal,
      resume: (...args) => calls.resume.push(args),
      block: (...args) => calls.block.push(args),
    },
    systemPrompt: { context: (context) => { calls.contexts.push(context); return () => {} } },
    effect: (fn) => fn(),
    on(event, fn) {
      const list = listeners.get(event) ?? []
      list.push(fn)
      listeners.set(event, list)
      return () => {}
    },
    emit(event, ...args) {
      for (const fn of listeners.get(event) ?? []) fn(...args)
    },
  }
  return {
    ctx,
    agent,
    session,
    goal,
    calls,
    listeners,
    emitSession(event) { ctx.emit('session/event', session, event) },
    emitIdle() { ctx.emit('agent/status', { agent, status: 'idle' }) },
  }
}

function reasoningMessage(turn, text) {
  return {
    type: 'assistant/message',
    data: {
      turn,
      step: 1,
      message: { role: 'assistant', content: [{ type: 'reasoning', text }] },
    },
  }
}

function textMessage(turn, text) {
  return {
    type: 'assistant/message',
    data: {
      turn,
      step: 1,
      message: { role: 'assistant', content: [{ type: 'text', text }] },
    },
  }
}

function turnEnd(turn, kind) {
  return { type: 'turn/end', data: { turn, reason: { kind } } }
}

function retryStart(turn, stage = 1) {
  return [
    {
      type: 'user/message',
      data: { source: { kind: 'reasoning-retry', form: 'notice', summary: `retry ${stage}/2` } },
    },
    { type: 'turn/start', data: { turn } },
  ]
}

test('isReasoningOnly distinguishes reasoning-only messages', () => {
  assert.equal(isReasoningOnly({ content: [{ type: 'reasoning', text: 'x' }] }), true)
  assert.equal(isReasoningOnly({ content: [{ type: 'reasoning', text: 'x' }, { type: 'text', text: 'y' }] }), false)
  assert.equal(isReasoningOnly({ content: [] }), false)
  assert.equal(isReasoningOnly(undefined), false)
})

test('repetitionStats finds a repeated n-gram', () => {
  const stats = repetitionStats('alpha beta gamma delta '.repeat(6), 4)
  assert.equal(stats.max, 6)
  assert.equal(stats.maxKey, 'alpha beta gamma delta')
  assert.ok(stats.ratio > 0)
})

test('the retry carries the lost reasoning and asks for visible progress, not less thinking', async () => {
  const h = makeHarness()
  apply(h.ctx, { maxRetries: 2 })
  h.emitSession(reasoningMessage(1, 'Option A needs a migration. ' + 'compare option a with option b again '.repeat(6)))
  h.emitSession(turnEnd(1, 'max-tokens'))
  h.emitIdle()
  await tick()

  assert.equal(h.calls.followups.length, 1)
  const retry = h.calls.followups[0]
  assert.equal(retry.source.kind, 'reasoning-retry')
  const text = retry.content[0].text
  assert.match(text, /<retry stage="1\/2">/)
  assert.match(text, /whole output budget \(32768 tokens\)/)
  assert.match(text, /<previous-reasoning-tail>\nOption A needs a migration/)
  assert.match(text, /Continue from it instead of starting over/)
  assert.match(text, /make progress visible/)
  assert.match(text, /You repeated "[a-z ]+" \d+ times\. Settle that point/)
  assert.doesNotMatch(text, /Do not think|reasoning off|do not re-analyze/i)
})

test('the reasoning tail is bounded', () => {
  const text = renderRetryMessage({ stage: 0, maxRetries: 2, reasoningTail: 'x'.repeat(10) })
  assert.match(text, /<previous-reasoning-tail>\nxxxxxxxxxx\n<\/previous-reasoning-tail>/)
})

test('step-by-step guidance applies to DeepSeek models only and names the output budget', () => {
  const h = makeHarness()
  apply(h.ctx, {})
  assert.equal(h.calls.contexts.length, 1)
  const context = h.calls.contexts[0]
  const text = context.text({ scope: h.agent })
  assert.match(text, /output budget per reply is 32768 tokens, and your reasoning counts toward it/)
  assert.match(text, /Work in steps/)
  const codex = makeHarness({ provider: 'openai-codex', model: 'gpt-6.1-sol' })
  apply(codex.ctx, {})
  assert.equal(codex.calls.contexts[0].text({ scope: codex.agent }), '')
  assert.match(renderGuidance(undefined), /output budget per reply is limited/)
})

test('escalates the prompt, then blocks and notifies after the retry budget', async () => {
  const h = makeHarness()
  apply(h.ctx, { maxRetries: 2 })
  h.emitSession(reasoningMessage(1, 'loop'))
  h.emitSession(turnEnd(1, 'max-tokens'))
  h.emitIdle()
  await tick()

  for (const event of retryStart(2)) h.emitSession(event)
  h.emitSession(reasoningMessage(2, 'still looping'))
  h.emitSession(turnEnd(2, 'max-tokens'))
  h.emitIdle()
  await tick()
  assert.equal(h.calls.followups.length, 2)
  assert.match(h.calls.followups[1].content[0].text, /stage="2\/2"/)
  assert.match(h.calls.followups[1].content[0].text, /split the work smaller/)
  assert.match(h.calls.followups[1].content[0].text, /still looping/)

  for (const event of retryStart(3, 2)) h.emitSession(event)

  h.emitSession(reasoningMessage(3, 'still looping'))
  h.emitSession(turnEnd(3, 'max-tokens'))
  await tick()

  assert.equal(h.calls.block.length, 1)
  assert.match(h.calls.block[0][2].message, /Output limit hit 2/)
  assert.equal(h.calls.followups.length, 3)
  assert.equal(h.calls.followups[2].source.kind, 'reasoning-retry-note')
})

test('a time-context message does not reset the retry ladder', async () => {
  const h = makeHarness()
  apply(h.ctx, { maxRetries: 2 })
  h.emitSession(reasoningMessage(1, 'loop'))
  h.emitSession(turnEnd(1, 'max-tokens'))
  h.emitIdle()
  await tick()

  for (const event of retryStart(2)) h.emitSession(event)
  h.emitSession({ type: 'user/message', data: { source: { kind: 'time-context', form: 'snapshot', sections: [] } } })
  h.emitSession(reasoningMessage(2, 'still looping'))
  h.emitSession(turnEnd(2, 'max-tokens'))
  h.emitIdle()
  await tick()

  assert.equal(h.calls.followups.length, 2)
  assert.match(h.calls.followups[1].content[0].text, /stage="2\/2"/)
})

test('give-up note is injected while the agent is still busy', async () => {
  const h = makeHarness()
  apply(h.ctx, { maxRetries: 2 })
  h.emitSession(reasoningMessage(1, 'loop'))
  h.emitSession(turnEnd(1, 'max-tokens'))
  h.emitIdle()
  await tick()

  for (const event of retryStart(2)) h.emitSession(event)
  h.emitSession(reasoningMessage(2, 'still looping'))
  h.emitSession(turnEnd(2, 'max-tokens'))
  h.emitIdle()
  await tick()

  for (const event of retryStart(3, 2)) h.emitSession(event)
  h.agent.status = 'running'
  h.emitSession(reasoningMessage(3, 'still looping'))
  h.emitSession(turnEnd(3, 'max-tokens'))
  await tick()

  assert.equal(h.calls.block.length, 1)
  assert.equal(h.calls.injected.length, 1)
  assert.equal(h.calls.injected[0].source.kind, 'reasoning-retry-note')
})

test('a successful retry resumes the disarmed goal', async () => {
  const h = makeHarness()
  apply(h.ctx, { maxRetries: 2 })
  h.emitSession(reasoningMessage(1, 'loop'))
  h.emitSession(turnEnd(1, 'max-tokens'))
  h.emitIdle()
  await tick()

  for (const event of retryStart(2)) h.emitSession(event)
  h.emitSession(textMessage(2, 'done'))
  h.emitSession(turnEnd(2, 'completed'))
  await tick()

  assert.equal(h.calls.resume.length, 1)
  assert.equal(h.calls.resume[0][1].id, 'goal-test')
})

test('a paused goal suppresses the retry', async () => {
  const h = makeHarness({ goalPhase: 'paused' })
  apply(h.ctx, { maxRetries: 2 })
  h.emitSession(reasoningMessage(1, 'loop'))
  h.emitSession(turnEnd(1, 'max-tokens'))
  h.emitIdle()
  await tick()
  assert.equal(h.calls.followups.length, 0)
})

test('completed reasoning-only retries when the goal is disarmed', async () => {
  const h = makeHarness({ goalActivation: 'disarmed' })
  apply(h.ctx, { maxRetries: 2 })
  h.emitSession(reasoningMessage(1, 'silent'))
  h.emitSession(turnEnd(1, 'completed'))
  h.emitIdle()
  await tick()
  assert.equal(h.calls.followups.length, 1)
})

test('completed reasoning-only defers to an armed goal round', async () => {
  const h = makeHarness({ goalActivation: 'armed' })
  apply(h.ctx, { maxRetries: 2 })
  h.emitSession(reasoningMessage(1, 'silent'))
  h.emitSession(turnEnd(1, 'completed'))
  h.emitIdle()
  await tick()
  assert.equal(h.calls.followups.length, 0)
})

test('a cap once forced below the adapter default is dropped so the default applies again', () => {
  const headers = [
    { config: { maxTokens: 32768 }, adapterDefaults: { maxTokens: true } },
    { config: { maxTokens: 8192 } },
    { config: { maxTokens: 4096 } },
  ]
  assert.equal(staleMaxTokens(headers, 4096), 32768)
  assert.equal(staleMaxTokens(headers, 32768), undefined)
  assert.equal(staleMaxTokens([{ config: { maxTokens: 4096 } }], 4096), undefined, 'no adapter default seen: leave it')
  assert.equal(staleMaxTokens(headers, undefined), undefined)
})
