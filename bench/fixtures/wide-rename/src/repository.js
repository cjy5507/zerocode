'use strict';

class Repository {
  constructor(records) {
    this.records = new Map(records.map((record) => [record.id, { ...record }]));
  }

  fetch(id) {
    return this.records.get(id) || null;
  }
}

module.exports = { Repository };
