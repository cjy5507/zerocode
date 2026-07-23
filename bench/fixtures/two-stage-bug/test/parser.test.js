'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const { parseConfig } = require('../src/parser');

test('parses trimmed keys and values before coercion', () => {
  assert.deepEqual(parseConfig(' port = 8080 \n enabled = true '), {
    port: 8080,
    enabled: true
  });
});

test('supports quoted strings containing equals signs', () => {
  assert.deepEqual(parseConfig('dsn = "postgres://u:p@example/db?ssl=true"'), {
    dsn: 'postgres://u:p@example/db?ssl=true'
  });
});

test('ignores whitespace-only and indented comment lines', () => {
  assert.deepEqual(parseConfig('\n  # local settings\nname = zo\n'), {
    name: 'zo'
  });
});

test('still rejects malformed non-empty lines', () => {
  assert.throws(() => parseConfig('missing delimiter'), /key=value/);
});
