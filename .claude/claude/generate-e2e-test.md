---
name: generate-e2e-test
description: Convert manual UI testing into a repeatable Playwright test
---

You've been manually testing the UI using the Playwright MCP tools. Now it's time to convert your testing session into a repeatable, automated test.

## Your Task

1. **Review your recent browser interactions** - Look at the MCP commands you've used (navigate, click, type, etc.)

2. **Create a Playwright test file** that reproduces your manual testing steps:
   - Name it descriptively: `e2e-tests/tests/claims-buddy/{feature}.spec.ts`
   - Include all the steps you took manually
   - Add appropriate assertions for what you observed
   - Handle timing/waiting appropriately (not just arbitrary waits)
   - Make the test deterministic and repeatable

3. **Document the test context** in comments:
   - What feature/flow you were testing
   - Any issues or edge cases you discovered
   - Why certain assertions or waits are necessary

4. **Verify the test works** by running it:
   ```bash
   docker run --rm --network llmsurance-network -v $(pwd)/e2e-tests:/app mcr.microsoft.com/playwright:v1.53.2-jammy sh -c "cd /app && npm test -- tests/claims-buddy/{your-test-file}.spec.ts"
   ```

5. **Fix any issues** found during verification:
   - Network configuration problems
   - Timing issues
   - Selector changes
   - Missing test data

6. **Report the results**:
   - Confirm the test passes
   - Explain any workarounds needed
   - Note any limitations or assumptions

## Example Output

```typescript
import { test, expect } from '@playwright/test';

test.describe('Feature: File Upload with Debug Mode', () => {
  test('should upload ESX file and display results with debug info', async ({ page }) => {
    // Test generated from manual session on [date]
    // Discovered: Network configuration issue when running in Docker
    // Workaround: Use host.docker.internal instead of localhost
    
    await page.goto('http://host.docker.internal:5173');
    
    // Enable debug mode first
    await page.getByRole('switch', { name: 'Debug' }).click();
    
    // ... rest of test
  });
});
```

Remember: The goal is to create a test that will reliably pass in CI/CD, not just a transcript of what you did manually.