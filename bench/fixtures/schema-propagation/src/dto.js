'use strict';

const { validateMoney } = require('./validate');

function toApiDto(money) {
  const errors = validateMoney(money);
  if (errors.length > 0) {
    throw new TypeError(errors.join('; '));
  }
  return {
    amount: money.amount,
    display: money.amount.toFixed(2)
  };
}

module.exports = { toApiDto };
