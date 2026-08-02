'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const { parseCsvChunks } = require('../src/parser');
const { splitRecords } = require('../src/tokenizer');

test('preserves existing unquoted CSV behavior across simple chunks', () => {
  assert.deepEqual(parseCsvChunks(['name,age\nAda,36\n', 'Grace,85']), [
    ['name', 'age'],
    ['Ada', '36'],
    ['Grace', '85']
  ]);
});

test('handles quoted commas when a field spans chunks', () => {
  assert.deepEqual(parseCsvChunks(['name,notes\r\nAda,"wrote, ', 'math"\r\n']), [
    ['name', 'notes'],
    ['Ada', 'wrote, math']
  ]);
});

test('handles escaped quotes split across chunk boundaries', () => {
  assert.deepEqual(parseCsvChunks(['id,text\n1,"He said ""hel', 'lo"""\n']), [
    ['id', 'text'],
    ['1', 'He said "hello"']
  ]);
});

test('keeps CRLF inside quoted fields while splitting records at real CRLF boundaries', () => {
  assert.deepEqual(parseCsvChunks(['id,body\r', '\n1,"two\r', '\nlines"\r\n2,done\r\n']), [
    ['id', 'body'],
    ['1', 'two\r\nlines'],
    ['2', 'done']
  ]);
});

test('tokenizer record splitting respects quotes rather than raw newlines', () => {
  assert.deepEqual(splitRecords(['a,b\n1,"x\n', 'y"\n2,z']), ['a,b', '1,"x\ny"', '2,z']);
});
