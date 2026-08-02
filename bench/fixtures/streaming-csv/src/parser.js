'use strict';

const { splitRecords, splitFields } = require('./tokenizer');

function parseCsvChunks(chunks) {
  return splitRecords(chunks).map((record) => splitFields(record));
}

module.exports = { parseCsvChunks };
