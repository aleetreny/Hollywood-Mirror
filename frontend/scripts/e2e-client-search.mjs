import {chromium} from 'playwright';

const browser = await chromium.launch({headless: true});
const page = await browser.newPage();
const diagnostics = [];

page.on('console', (message) => {
  diagnostics.push(`console.${message.type()}: ${message.text()}`);
});
page.on('pageerror', (error) => diagnostics.push(`pageerror: ${error.message}`));
page.on('requestfailed', (request) => {
  diagnostics.push(`requestfailed: ${request.url()} (${request.failure()?.errorText ?? 'unknown'})`);
});
page.on('response', (response) => {
  if (response.status() >= 400) {
    diagnostics.push(`response: ${response.status()} ${response.url()}`);
  }
});

let failure;

try {
  await page.goto('http://127.0.0.1:4173', {
    waitUntil: 'domcontentloaded',
    timeout: 60_000,
  });

  const prompt = page.getByLabel('Your Idea or Script Fragment');
  await prompt.fill(
    'A lonely astronaut searches for signs of life after losing contact with Earth.',
  );

  const searchButton = page.getByRole('button', {name: 'Find similar movies'});
  if (!(await searchButton.isEnabled())) {
    throw new Error('The search button is disabled before the first search.');
  }

  await searchButton.click();
  await page.locator('h2:has-text("Top Matches"), h3:has-text("Search Failed")').waitFor({
    timeout: 240_000,
  });

  const searchFailed = page.getByRole('heading', {name: 'Search Failed'});
  if (await searchFailed.isVisible()) {
    const failurePanel = searchFailed.locator('..');
    throw new Error(`The browser search failed: ${await failurePanel.innerText()}`);
  }

  const resultRows = page.locator('main .divide-y > div');
  const resultCount = await resultRows.count();
  if (resultCount !== 5) {
    throw new Error(`Expected 5 movie results, received ${resultCount}.`);
  }

  const mainText = await page.locator('main').innerText();
  if (!mainText.includes('Private local search ready')) {
    throw new Error('The local search engine did not reach the ready state.');
  }

  if (diagnostics.some((entry) => entry.startsWith('pageerror:'))) {
    throw new Error('The page emitted an uncaught browser error.');
  }

  console.log(`Client-side semantic search passed with ${resultCount} results.`);
} catch (error) {
  failure = error;
  diagnostics.push(`test-error: ${error instanceof Error ? error.stack ?? error.message : String(error)}`);
  diagnostics.push(`url: ${page.url()}`);
  diagnostics.push(
    `main-text:\n${await page.locator('main').innerText().catch(() => '<main unavailable>')}`,
  );
} finally {
  await page.screenshot({path: 'client-search-e2e.png', fullPage: true}).catch(() => undefined);
  if (diagnostics.length > 0) console.error(diagnostics.join('\n'));
  await browser.close();
}

if (failure) throw failure;
