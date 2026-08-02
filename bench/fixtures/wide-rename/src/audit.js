'use strict';

function describeLookup(repository, id) {
  const user = repository.fetch(id);
  if (!user) {
    return `missing:${id}`;
  }
  return `loaded:${user.id}:${user.email}`;
}

module.exports = { describeLookup };
