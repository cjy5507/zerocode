'use strict';

function splitRecords(chunks) {
  return chunks
    .join('')
    .split(/\r?\n/)
    .filter((line) => line.length > 0);
}

function splitFields(record) {
  return record.split(',');
}

module.exports = { splitRecords, splitFields };
