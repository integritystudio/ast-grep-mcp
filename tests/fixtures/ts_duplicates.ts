// TypeScript fixture with known-duplicate functions for recall regression tests.
// Functions processUserData and processOrderData are structurally near-identical
// (same shape, typed params, different domain names) — a deduplication tool
// should report them as a duplication group.

interface User {
  id: number;
  name: string;
  email: string;
}

interface Order {
  id: number;
  customerId: number;
  total: number;
}

function processUserData(user: User): string {
  const validated = validateInput(user.id);
  if (!validated) {
    throw new Error("Invalid user id");
  }
  const result = transformData(user.name, user.email);
  logOperation("process_user", user.id);
  return result;
}

function processOrderData(order: Order): string {
  const validated = validateInput(order.id);
  if (!validated) {
    throw new Error("Invalid order id");
  }
  const result = transformData(order.customerId.toString(), order.total.toString());
  logOperation("process_order", order.id);
  return result;
}

function validateInput(id: number): boolean {
  return id > 0;
}

function transformData(a: string, b: string): string {
  return `${a}:${b}`;
}

function logOperation(op: string, id: number): void {
  console.log(`${op} id=${id}`);
}
