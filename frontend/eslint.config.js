import tseslint from 'typescript-eslint';

export default tseslint.config(
  { ignores: ['dist/**', 'node_modules/**', 'public/**', '*.tsbuildinfo'] },
  ...tseslint.configs.recommended,
  // The existing market transport decodes an untyped public WebSocket response.
  { files: ['src/lib/deriv.ts', 'tests/market_session.test.ts'], rules: { '@typescript-eslint/no-explicit-any': 'warn' } },
);
