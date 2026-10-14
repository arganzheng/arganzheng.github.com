(function () {
  var cachedAt = document.documentElement.dataset.swOffline;
  if (!cachedAt || !document.body) return;

  var date = new Date(Number(cachedAt));
  if (Number.isNaN(date.getTime())) return;
  function pad(value) { return String(value).padStart(2, '0'); }
  var timestamp = date.getFullYear() + '-' + pad(date.getMonth() + 1) + '-' +
    pad(date.getDate()) + ' ' + pad(date.getHours()) + ':' + pad(date.getMinutes());

  var banner = document.createElement('div');
  banner.className = 'offline-banner';
  banner.setAttribute('role', 'status');
  banner.setAttribute('aria-live', 'polite');

  var message = document.createElement('span');
  message.className = 'offline-banner-message';
  message.textContent = '离线版本 · 缓存于 ' + timestamp + ' · 联网后刷新获取最新';

  var refresh = document.createElement('button');
  refresh.type = 'button';
  refresh.className = 'offline-banner-refresh';
  refresh.textContent = '刷新';
  refresh.addEventListener('click', function () { location.reload(); });

  banner.appendChild(message);
  banner.appendChild(refresh);
  document.body.insertBefore(banner, document.body.firstChild);
  document.body.classList.add('has-offline-banner');

  function updateNavbarOffset() {
    document.documentElement.style.setProperty('--offline-banner-height', banner.offsetHeight + 'px');
  }
  updateNavbarOffset();
  window.addEventListener('resize', updateNavbarOffset, { passive: true });
})();
