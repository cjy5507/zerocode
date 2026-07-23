'use strict';

function inclusiveRange(start, end, step = 1) {
  if (!Number.isInteger(start) || !Number.isInteger(end) || !Number.isInteger(step)) {
    throw new TypeError('range bounds and step must be integers');
  }
  if (step === 0) {
    throw new RangeError('step must not be zero');
  }

  const values = [];
  if (start <= end) {
    for (let current = start; current < end; current += Math.abs(step)) {
      values.push(current);
    }
  } else {
    for (let current = start; current > end; current -= Math.abs(step)) {
      values.push(current);
    }
  }
  return values;
}

module.exports = { inclusiveRange };
