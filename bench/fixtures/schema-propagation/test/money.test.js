'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const { createMoney } = require('../src/model');
const { serializeMoney, deserializeMoney } = require('../src/serialize');
const { validateMoney } = require('../src/validate');
const { toApiDto } = require('../src/dto');

test('creates immutable money with an explicit currency', () => {
  const money = createMoney(12.5, 'USD');
  assert.deepEqual(money, { amount: 12.5, currency: 'USD' });
  assert.throws(() => {
    money.currency = 'EUR';
  });
});

test('requires a three-letter uppercase currency code', () => {
  assert.throws(() => createMoney(3, 'usd'), /currency/);
  assert.throws(() => createMoney(3, 'US'), /currency/);
  assert.throws(() => createMoney(3, ''), /currency/);
});

test('serializes and deserializes currency end-to-end', () => {
  const money = createMoney(7, 'EUR');
  const json = serializeMoney(money);
  assert.equal(json, '{"amount":7,"currency":"EUR"}');
  assert.deepEqual(deserializeMoney(json), { amount: 7, currency: 'EUR' });
});

test('validates currency wherever money enters the API layer', () => {
  assert.deepEqual(validateMoney({ amount: 2, currency: 'JPY' }), []);
  assert.deepEqual(validateMoney({ amount: 2 }), ['currency must be a three-letter uppercase code']);
  assert.deepEqual(validateMoney({ amount: 2, currency: 'yen' }), ['currency must be a three-letter uppercase code']);
});

test('threads currency into the public DTO without changing display formatting', () => {
  assert.deepEqual(toApiDto(createMoney(9, 'GBP')), {
    amount: 9,
    currency: 'GBP',
    display: '9.00 GBP'
  });
});
