'use strict';

class Store {
  constructor(entries = {}) {
    this.values = new Map(Object.entries(entries));
  }

  get(key) {
    return this.values.get(key);
  }

  set(key, value) {
    this.values.set(key, value);
  }

  delete(key) {
    this.values.delete(key);
  }

  toObject() {
    return Object.fromEntries([...this.values.entries()].sort(([left], [right]) => left.localeCompare(right)));
  }
}

module.exports = { Store };
