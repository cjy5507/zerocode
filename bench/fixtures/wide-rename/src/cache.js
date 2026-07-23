'use strict';

function cachedUser(repository, id, cache) {
  if (!cache.has(id)) {
    cache.set(id, repository.fetch(id));
  }
  return cache.get(id);
}

module.exports = { cachedUser };
