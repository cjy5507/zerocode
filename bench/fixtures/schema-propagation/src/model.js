'use strict';

function createMoney(amount) {
  if (!Number.isFinite(amount)) {
    throw new TypeError('amount must be a finite number');
  }
  return Object.freeze({ amount });
}

module.exports = { createMoney };
