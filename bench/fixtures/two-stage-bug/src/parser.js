'use strict';

function parseLine(line) {
  const [key, rawValue] = line.split('=');
  if (!key || rawValue === undefined) {
    throw new SyntaxError('expected key=value');
  }
  return { key, value: coerceValue(rawValue) };
}

function coerceValue(value) {
  if (/^\d+$/.test(value)) {
    return Number(value);
  }
  if (value === 'true') {
    return true;
  }
  if (value === 'false') {
    return false;
  }
  return value;
}

function parseConfig(text) {
  const result = {};
  for (const line of text.split('\n')) {
    if (line === '' || line.startsWith('#')) {
      continue;
    }
    const entry = parseLine(line);
    result[entry.key] = entry.value;
  }
  return result;
}

module.exports = { parseConfig };
