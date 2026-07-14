import {chromium} from 'playwright';

const browser = await chromium.launch({headless: true});
const page = await browser.newPage();
const browserErrors = [];

page.on('pageerror', (error) => browserErrors.push(`pageerror: ${error.message}`));
page.on('requestfailed', (request) => {
  browserErrors.push(`requestfailed: ${request.url()} (${request.failure()?.errorText ?? 'unknown'})`);
});

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
  await page.getByRole('heading', {name: 'Top Matches'}).waitFor({timeout: 240_000});

  const resultRows = page.locator('main .divide-y > div');
  const resultCount = await resultRows.count();
  if (resultCount !== 5) {
    throw new Error(`Expected 5 movie results, received ${resultCount}.`);
  }

  const mainText = await page.locator('main').innerText();
  if (!mainText.includes('Private local search ready')) {
    throw new Error('The local search engine did not reach the ready state.');
  }

  if (browserErrors.length > 0) {
    throw new Error(browserErrors.join('\n'));
  }

  await page.screenshot({path: 'client-search-e2e.png', fullPage: true});
  console.log(`Client-side semantic search passed with ${resultCount} results.`);
} finally {
  await browser.close();
}
