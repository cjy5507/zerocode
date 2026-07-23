'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const { applyBatch } = require('../src/batch');
const { BatchError } = require('../src/errors');
const { Store } = require('../src/store');

function handlers(failAtKey = null) {
  return {
    async put(store, operation) {
      await Promise.resolve();
      if (operation.key === failAtKey) {
        throw new Error(`boom:${operation.key}`);
      }
      store.set(operation.key, operation.value);
      return `put:${operation.key}`;
    },
    async remove(store, operation) {
      await Promise.resolve();
      if (operation.key === failAtKey) {
        throw new Error(`boom:${operation.key}`);
      }
      store.delete(operation.key);
      return `remove:${operation.key}`;
    }
  };
}

test('commits all async operations in order when every handler succeeds', async () => {
  const store = new Store({ a: 'old', b: 'keep' });
  const result = await applyBatch(store, [
    { type: 'put', key: 'a', value: 'new' },
    { type: 'put', key: 'c', value: 'created' },
    { type: 'remove', key: 'b' }
  ], handlers());

  assert.deepEqual(result, ['put:a', 'put:c', 'remove:b']);
  assert.deepEqual(store.toObject(), { a: 'new', c: 'created' });
});

test('rolls back every prior write when a later async operation rejects', async () => {
  const store = new Store({ a: 'old', b: 'keep' });

  await assert.rejects(
    () => applyBatch(store, [
      { type: 'put', key: 'a', value: 'new' },
      { type: 'put', key: 'c', value: 'created' },
      { type: 'remove', key: 'b' }
    ], handlers('c')),
    (error) => {
      assert.equal(error instanceof BatchError, true);
      assert.equal(error.operationIndex, 1);
      assert.equal(error.operation.key, 'c');
      assert.match(error.cause.message, /boom:c/);
      return true;
    }
  );

  assert.deepEqual(store.toObject(), { a: 'old', b: 'keep' });
});

test('rolls back deletes as well as writes', async () => {
  const store = new Store({ a: 'old', b: 'keep', c: 'stay' });

  await assert.rejects(
    () => applyBatch(store, [
      { type: 'remove', key: 'b' },
      { type: 'put', key: 'a', value: 'new' },
      { type: 'put', key: 'z', value: 'fail' }
    ], handlers('z')),
    BatchError
  );

  assert.deepEqual(store.toObject(), { a: 'old', b: 'keep', c: 'stay' });
});

test('missing handlers are reported as rollback-safe BatchError instances', async () => {
  const store = new Store({ a: 'old' });

  await assert.rejects(
    () => applyBatch(store, [
      { type: 'put', key: 'a', value: 'new' },
      { type: 'unknown', key: 'x', value: 'bad' }
    ], handlers()),
    (error) => {
      assert.equal(error instanceof BatchError, true);
      assert.equal(error.operationIndex, 1);
      assert.equal(error.operation.type, 'unknown');
      return true;
    }
  );

  assert.deepEqual(store.toObject(), { a: 'old' });
});
