// Detects reasoning that has collapsed into repeating the same phrases.
//
// Novelty is measured on word n-grams: in ordinary reasoning nearly every
// 8-word sequence is new, while a degenerate loop repeats a handful of them
// thousands of times. The point where novelty stays low marks the start of the
// loop; the text before it is the part still worth keeping.

const GRAM = 8

/**
 * @param text - reasoning text so far.
 * @param options.minWords - do not judge texts shorter than this.
 * @param options.window - words in the trailing window that must be repetitive.
 * @param options.maxNovelty - highest share of new n-grams that still counts as a loop.
 * @returns undefined, or where the loop starts, the text before it, and the phrases it repeats.
 */
export function detectDegeneration(text, { minWords = 400, window = 300, maxNovelty = 0.15 } = {}) {
  const words = [...String(text ?? '').matchAll(/\S+/g)]
  if (words.length < Math.max(minWords, window + GRAM)) return undefined
  const fresh = new Uint8Array(words.length)
  const seen = new Set()
  for (let i = GRAM - 1; i < words.length; i++) {
    let gram = ''
    for (let j = i - GRAM + 1; j <= i; j++) gram += words[j][0] + ' '
    if (!seen.has(gram)) { fresh[i] = 1; seen.add(gram) }
  }
  const prefix = new Uint32Array(words.length + 1)
  for (let i = 0; i < words.length; i++) prefix[i + 1] = prefix[i] + fresh[i]
  const novelty = (from, to) => (prefix[to] - prefix[from]) / (to - from)
  if (novelty(words.length - window, words.length) > maxNovelty) return undefined
  // Walk back to the last stretch of new text; the loop starts after it.
  const span = 100
  let start = 0
  for (let from = words.length - span; from >= 0; from -= 10) {
    if (novelty(from, from + span) > 0.5) { start = from + span; break }
  }
  let startChar = words[start]?.index ?? text.length
  // The first rounds of a loop still look new; move back to the first occurrence of
  // its most repeated phrase within a few thousand characters before the boundary.
  const top = repeatedPhrases(text.slice(startChar), 1)[0]
  if (top !== undefined && top.phrase.length >= 15) {
    const first = text.indexOf(top.phrase, Math.max(0, startChar - 4000))
    if (first >= 0 && first < startChar) startChar = first
  }
  return {
    start: startChar,
    healthy: text.slice(0, startChar).trimEnd(),
    loopChars: text.length - startChar,
    phrases: repeatedPhrases(text.slice(startChar)),
  }
}

/** The most repeated sentence pieces of a loop, most frequent first. */
export function repeatedPhrases(loop, limit = 3) {
  const counts = new Map()
  for (const piece of loop.split(/[.;?!](?:\s+|$)|\n+/)) {
    const phrase = piece.trim()
    if (phrase.length < 3) continue
    counts.set(phrase, (counts.get(phrase) ?? 0) + 1)
  }
  return [...counts].filter(([, n]) => n >= 3).sort((a, b) => b[1] - a[1]).slice(0, limit).map(([phrase, count]) => ({ phrase: phrase.slice(0, 120), count }))
}
