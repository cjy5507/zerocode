'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const { Repository } = require('../src/repository');
const { renderUser } = require('../src/controller');

function sampleRepository() {
  return new Repository([
    { id: 'u1', firstName: 'Ada', lastName: 'Lovelace', email: 'ada@example.com', deleted: false },
    { id: 'u2', firstName: 'Grace', lastName: 'Hopper', email: 'grace@example.com', deleted: true }
  ]);
}

test('Repository.load replaces fetch and preserves existing callers through opts', () => {
  const repository = sampleRepository();
  assert.equal(typeof repository.fetch, 'undefined');
  assert.deepEqual(repository.load('u1'), {
    id: 'u1',
    firstName: 'Ada',
    lastName: 'Lovelace',
    email: 'ada@example.com',
    deleted: false
  });
  assert.equal(repository.load('u2'), null);
  assert.equal(repository.load('u2', { includeDeleted: true }).email, 'grace@example.com');
});

test('all call sites thread opts through the renamed API', () => {
  const repository = sampleRepository();
  const visible = renderUser(repository, 'u1');
  assert.equal(visible.name, 'Ada Lovelace');
  assert.equal(visible.email, 'ada@example.com');
  assert.equal(visible.audit, 'loaded:u1:ada@example.com');
  assert.equal(visible.cached.email, 'ada@example.com');

  const deleted = renderUser(repository, 'u2');
  assert.equal(deleted.name, null);
  assert.equal(deleted.email, null);
  assert.equal(deleted.audit, 'missing:u2');
  assert.equal(deleted.cached, null);
}
);
