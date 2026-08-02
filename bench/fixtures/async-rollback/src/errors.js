'use strict';

class BatchError extends Error {
  constructor(message) {
    super(message);
    this.name = 'BatchError';
  }
}

module.exports = { BatchError };
