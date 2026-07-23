'use strict';

const { getDisplayName, getEmail } = require('./service');
const { describeLookup } = require('./audit');
const { cachedUser } = require('./cache');

function renderUser(repository, id, cache = new Map()) {
  return {
    name: getDisplayName(repository, id),
    email: getEmail(repository, id),
    audit: describeLookup(repository, id),
    cached: cachedUser(repository, id, cache)
  };
}

module.exports = { renderUser };
