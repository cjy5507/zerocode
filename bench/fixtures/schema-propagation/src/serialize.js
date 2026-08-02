'use strict';

function serializeMoney(money) {
  return JSON.stringify({ amount: money.amount });
}

function deserializeMoney(json) {
  const parsed = JSON.parse(json);
  return { amount: parsed.amount };
}

module.exports = { serializeMoney, deserializeMoney };
