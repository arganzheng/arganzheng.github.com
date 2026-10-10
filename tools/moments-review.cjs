#!/usr/bin/env node
'use strict';

const { createHash } = require('node:crypto');
const { pickReview } = require('../js/moments-review.js');

const DEFAULT_SITE_URL = 'https://arganzheng.life';
const INDEX_PATH = '/moments/index.json';

function absoluteUrl(value, siteUrl) {
  return new URL(value, siteUrl.endsWith('/') ? siteUrl : `${siteUrl}/`).href;
}

function buildMessage(entries, siteUrl = DEFAULT_SITE_URL) {
  const title = `随笔回顾 · ${entries[0].label}${entries.length > 1 ? `（共 ${entries.length} 条）` : ''}`;
  const body = entries.map(({ e, label }) => {
    const dateTime = [e.date, e.time].filter(Boolean).join(' ');
    const heading = [`**${label}**`, dateTime, e.place].filter(Boolean).join(' · ');
    const text = String(e.text || '');
    const excerpt = text.length > 500 ? `${text.slice(0, 499)}…` : text;
    const image = e.img ? `![](${absoluteUrl(e.img, siteUrl)})` : '';
    const link = `[查看原文](${absoluteUrl(e.url, siteUrl)})`;
    return [heading, excerpt, image, link].filter(Boolean).join('\n\n');
  }).join('\n\n---\n\n');
  return { title, body };
}

function pickFallback(all, date) {
  if (!all.length) return [];
  const index = createHash('sha256').update(date).digest().readUInt32BE(0) % all.length;
  return [{ e: all[index], label: '随机回顾' }];
}

function serverChanEndpoint(sendkey) {
  const match = /^sctp(\d+)t/i.exec(sendkey);
  return match
    ? `https://${match[1]}.push.ft07.com/send/${sendkey}.send`
    : `https://sctapi.ftqq.com/${sendkey}.send`;
}

function getReviewDate(now = new Date()) {
  if (process.env.REVIEW_DATE) {
    const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(process.env.REVIEW_DATE);
    if (!match) throw new Error('REVIEW_DATE must use YYYY-MM-DD');
    const [year, month, day] = match.slice(1).map(Number);
    const parsed = new Date(Date.UTC(year, month - 1, day));
    if (parsed.getUTCFullYear() !== year || parsed.getUTCMonth() !== month - 1 || parsed.getUTCDate() !== day) {
      throw new Error('REVIEW_DATE must be a valid calendar date');
    }
    return { date: process.env.REVIEW_DATE, year, month, day };
  }
  const parts = new Intl.DateTimeFormat('en', {
    timeZone: 'Asia/Shanghai',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit'
  }).formatToParts(now);
  const dateParts = Object.fromEntries(parts.map(part => [part.type, part.value]));
  return { date: `${dateParts.year}-${dateParts.month}-${dateParts.day}`, year: Number(dateParts.year), month: Number(dateParts.month), day: Number(dateParts.day) };
}

function safeMessage(value, secrets) {
  let message = String(value || '');
  for (const secret of secrets.filter(Boolean)) message = message.split(secret).join('[redacted]');
  return message;
}

async function readJson(response) {
  const text = await response.text();
  try {
    return JSON.parse(text);
  } catch {
    return { msg: text };
  }
}

async function main() {
  const serverChanKey = process.env.SERVERCHAN_SENDKEY || '';
  const pushPlusToken = process.env.PUSHPLUS_TOKEN || '';
  const secrets = [serverChanKey, pushPlusToken];
  const dryRun = process.argv.includes('--dry-run') || /^(1|true)$/i.test(process.env.DRY_RUN || '');

  if (!dryRun && !serverChanKey && !pushPlusToken) {
    console.log('no push channel configured');
    return 0;
  }

  try {
    const siteUrl = (process.env.SITE_URL || DEFAULT_SITE_URL).replace(/\/+$/, '');
    const response = await fetch(`${siteUrl}${INDEX_PATH}`);
    if (!response.ok) throw new Error(`Moments index returned HTTP ${response.status}`);
    const all = await response.json();
    const { date, year, month, day } = getReviewDate();
    const entries = pickReview(all, year, month, day);
    const selected = entries.length ? entries : pickFallback(all, date);
    if (!selected.length) throw new Error('Moments index contains no entries');
    const message = buildMessage(selected, siteUrl);

    if (dryRun) {
      console.log(`${message.title}\n\n${message.body}`);
      return 0;
    }

    let failed = false;
    if (serverChanKey) {
      try {
        const form = new URLSearchParams({ title: message.title, desp: message.body });
        const result = await fetch(serverChanEndpoint(serverChanKey), {
          method: 'POST',
          headers: { 'content-type': 'application/x-www-form-urlencoded' },
          body: form
        });
        const data = await readJson(result);
        if (!result.ok || data.code !== 0) {
          failed = true;
          console.error(`Server酱 failed: ${safeMessage(data.msg || `HTTP ${result.status}`, secrets)}`);
        } else {
          console.log(`Server酱: ${safeMessage(data.msg || 'success', secrets)}`);
        }
      } catch (error) {
        failed = true;
        console.error(`Server酱 failed: ${safeMessage(error.message, secrets)}`);
      }
    }
    if (pushPlusToken) {
      try {
        const result = await fetch('https://www.pushplus.plus/send', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ token: pushPlusToken, title: message.title, content: message.body, template: 'markdown' })
        });
        const data = await readJson(result);
        if (!result.ok || data.code !== 200) {
          failed = true;
          console.error(`PushPlus failed: ${safeMessage(data.msg || `HTTP ${result.status}`, secrets)}`);
        } else {
          console.log(`PushPlus: ${safeMessage(data.msg || 'success', secrets)}`);
        }
      } catch (error) {
        failed = true;
        console.error(`PushPlus failed: ${safeMessage(error.message, secrets)}`);
      }
    }
    return failed ? 1 : 0;
  } catch (error) {
    console.error(safeMessage(error.message, secrets));
    return 1;
  }
}

if (require.main === module) {
  main().then(code => { process.exitCode = code; }).catch(error => {
    console.error(safeMessage(error.message, [process.env.SERVERCHAN_SENDKEY, process.env.PUSHPLUS_TOKEN]));
    process.exitCode = 1;
  });
}

module.exports = { buildMessage, pickFallback, serverChanEndpoint };
