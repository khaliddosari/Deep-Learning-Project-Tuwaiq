// Bindings exposed to the Worker at runtime.
//
// `@cloudflare/workers-types` declares an empty `Cloudflare.Env`; TypeScript
// merges this declaration with it. (`wrangler types` would generate this file
// automatically for a project with a wrangler config.)
//
// `DB` is optional because it is only provisioned when `.openai/hosting.json`
// sets `"d1": "DB"`. `db/getDb()` throws a clear error when it is absent, so the
// optional type matches the runtime guard rather than hiding it.
//
// No top-level import/export: this file must stay a global script so the
// namespace declaration merges instead of becoming module-local.
declare namespace Cloudflare {
  interface Env {
    DB?: D1Database;
  }
}
