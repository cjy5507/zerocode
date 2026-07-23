'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const { inclusiveRange } = require('../src/range');

test('builds ascending inclusive ranges', () => {
  assert.deepEqual(inclusiveRange(2, 5), [2, 3, 4, 5]);
});

test('builds descending inclusive ranges', () => {
  assert.deepEqual(inclusiveRange(5, 2), [5, 4, 3, 2]);
});

test('honors explicit step while keeping the end inclusive when aligned', () => {
  assert.deepEqual(inclusiveRange(2, 10, 2), [2, 4, 6, 8, 10]);
});

test('rejects zero step', () => {
  assert.throws(() => inclusiveRange(1, 5, 0), /step/);
});
