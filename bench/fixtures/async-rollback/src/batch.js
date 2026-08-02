'use strict';

const { BatchError } = require('./errors');

async function applyBatch(store, operations, handlers) {
  const results = [];
  for (const operation of operations) {
    const handler = handlers[operation.type];
    if (!handler) {
      throw new BatchError(`missing handler for ${operation.type}`);
    }
    const value = await handler(store, operation);
    results.push(value);
  }
  return results;
}

module.exports = { applyBatch };
