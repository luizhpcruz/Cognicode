import { myFunction } from '../src/index';

describe('My Function Tests', () => {
  test('should return expected result for input 1', () => {
    const result = myFunction(1);
    expect(result).toBe('expected result for input 1');
  });

  test('should return expected result for input 2', () => {
    const result = myFunction(2);
    expect(result).toBe('expected result for input 2');
  });

  test('should handle edge case', () => {
    const result = myFunction(0);
    expect(result).toBe('expected result for edge case');
  });
});