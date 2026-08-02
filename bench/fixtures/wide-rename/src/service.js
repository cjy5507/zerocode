'use strict';

function getDisplayName(repository, id) {
  const user = repository.fetch(id);
  return user ? `${user.firstName} ${user.lastName}` : null;
}

function getEmail(repository, id) {
  const user = repository.fetch(id);
  return user ? user.email : null;
}

module.exports = { getDisplayName, getEmail };
