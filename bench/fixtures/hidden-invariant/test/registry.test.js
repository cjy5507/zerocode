'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const { Registry } = require('../src/registry');

test('remove deletes one item and reports whether anything changed', () => {
  const registry = new Registry();
  const low = registry.add('low', 1);
  const high = registry.add('high', 10);

  assert.equal(registry.remove(low), true);
  assert.equal(registry.get(low), null);
  assert.deepEqual(registry.list().map((item) => item.id), [high]);
  assert.equal(registry.remove(low), false);
});

test('hidden invariant: removing an item must not disturb priority ordering', () => {
  const registry = new Registry();
  const medium = registry.add('medium', 5);
  const high = registry.add('high', 10);
  const alsoHigh = registry.add('also-high', 10);
  const low = registry.add('low', 1);

  assert.deepEqual(registry.list().map((item) => item.id), [high, alsoHigh, medium, low]);
  assert.equal(registry.remove(medium), true);
  assert.deepEqual(registry.list().map((item) => item.id), [high, alsoHigh, low]);
});

test('hidden invariant: ids are never reused after removal', () => {
  const registry = new Registry();
  const first = registry.add('first', 0);
  registry.remove(first);
  const second = registry.add('second', 0);
  assert.equal(second, first + 1);
});

test('list remains defensive after removals', () => {
  const registry = new Registry();
  const id = registry.add('owned', 3);
  const listed = registry.list();
  listed[0].name = 'mutated';
  assert.equal(registry.get(id).name, 'owned');
});
