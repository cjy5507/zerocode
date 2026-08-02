'use strict';

class Registry {
  constructor() {
    this.items = [];
    this.nextId = 1;
  }

  add(name, priority = 0) {
    const item = { id: this.nextId++, name, priority };
    this.items.push(item);
    this.items.sort((left, right) => {
      if (right.priority !== left.priority) {
        return right.priority - left.priority;
      }
      return left.id - right.id;
    });
    return item.id;
  }

  list() {
    return this.items.map((item) => ({ ...item }));
  }

  get(id) {
    return this.items.find((item) => item.id === id) || null;
  }
}

module.exports = { Registry };
