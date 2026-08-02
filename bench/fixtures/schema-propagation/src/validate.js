'use strict';

function validateMoney(value) {
  const errors = [];
  if (typeof value !== 'object' || value === null) {
    return ['money must be an object'];
  }
  if (!Number.isFinite(value.amount)) {
    errors.push('amount must be a finite number');
  }
  return errors;
}

module.exports = { validateMoney };
