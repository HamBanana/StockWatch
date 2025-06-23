// Requires: puppeteer
// Install with: npm install puppeteer

const puppeteer = require('puppeteer');
const fs = require('fs');

const ticker = process.argv[2];
if (!ticker) {
  console.error('Please provide a ticker symbol as an argument.');
  process.exit(1);
}

(async () => {
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();

  console.log(`Navigating to Yahoo Finance page for ${ticker} historical data...`);
  await page.goto(`https://finance.yahoo.com/quote/${ticker}/history`, { waitUntil: 'networkidle2' });

  // Accept cookies if popup appears
  try {
    await page.waitForSelector('button[name="agree"]', { timeout: 3000 });
    await page.click('button[name="agree"]');
  } catch (err) {}

  // Wait for historical table to load
  await page.waitForSelector('table[data-test="historical-prices"]');

  // Scrape historical OHLCV data
  const ohlcvData = await page.evaluate(() => {
    const rows = document.querySelectorAll('table[data-test="historical-prices"] tbody tr');
    const extracted = [];

    for (let row of rows) {
      const cols = row.querySelectorAll('td');
      if (cols.length < 6) continue;

      const date = cols[0].innerText.trim();
      const open = parseFloat(cols[1].innerText.replace(/,/g, ''));
      const high = parseFloat(cols[2].innerText.replace(/,/g, ''));
      const low = parseFloat(cols[3].innerText.replace(/,/g, ''));
      const close = parseFloat(cols[4].innerText.replace(/,/g, ''));
      const adjClose = parseFloat(cols[5].innerText.replace(/,/g, ''));
      const volume = parseInt(cols[6].innerText.replace(/,/g, ''));

      extracted.push({ date, open, high, low, close, adjClose, volume });
    }

    return extracted;
  });

  console.log(`Navigating to Yahoo Finance statistics page for ${ticker}...`);
  await page.goto(`https://finance.yahoo.com/quote/${ticker}/key-statistics`, { waitUntil: 'networkidle2' });

  // Scrape technical indicators (as available on stats page)
  const indicators = await page.evaluate(() => {
    const findValue = (label) => {
      const row = Array.from(document.querySelectorAll('tr')).find(r => r.innerText.includes(label));
      if (!row) return null;
      const val = row.querySelector('td:nth-child(2)')?.innerText;
      return val || null;
    };

    return {
      marketCap: findValue('Market Cap'),
      beta: findValue('Beta'),
      PE: findValue('Trailing P/E'),
      EPS: findValue('Diluted EPS'),
      forwardPE: findValue('Forward P/E'),
      priceToBook: findValue('Price/Book'),
      profitMargin: findValue('Profit Margin'),
      returnOnAssets: findValue('Return on Assets'),
      returnOnEquity: findValue('Return on Equity'),
      totalCash: findValue('Total Cash'),
      totalDebt: findValue('Total Debt'),
      currentRatio: findValue('Current Ratio'),
      leveredFreeCashFlow: findValue('Levered Free Cash Flow')
    };
  });

  console.log(`Navigating to Yahoo Finance analysis page for ${ticker}...`);
  await page.goto(`https://finance.yahoo.com/quote/${ticker}/analysis`, { waitUntil: 'networkidle2' });

  // Scrape analysis-based metrics (estimates, trends, etc.)
  const analysis = await page.evaluate(() => {
    const data = {};
    const rows = document.querySelectorAll('section[data-test="qsp-analyst"] table tr');

    rows.forEach(row => {
      const cells = row.querySelectorAll('td');
      if (cells.length > 1) {
        const label = cells[0].innerText.trim();
        const value = cells[1].innerText.trim();
        data[label] = value;
      }
    });

    return data;
  });

  const output = {
    ticker,
    indicators,
    analysis,
    data: ohlcvData
  };

  const outputFile = `./${ticker}_full_technical_data.json`;
  fs.writeFileSync(outputFile, JSON.stringify(output, null, 2));
  console.log(`Full technical and analysis data saved to ${outputFile}`);

  await browser.close();
})();
