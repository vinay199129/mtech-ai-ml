/* FR-009: GoatCounter cookieless analytics bootstrap (shared by all pages).
   Replace YOUR-SITE in window.SITE_GC_URL below with your GoatCounter site code
   after signing up at https://www.goatcounter.com/. Until that is done, this
   script is inert (no network requests fire). */

window.SITE_GC_URL = 'https://YOUR-SITE.goatcounter.com/count';
window.goatcounter = window.goatcounter || { no_onload: true };

(function () {
  var s = document.createElement('script');
  s.async = true;
  s.src = '//gc.zgo.at/count.js';
  s.setAttribute('data-goatcounter', window.SITE_GC_URL);
  document.head.appendChild(s);

  function isConfigured() {
    return window.SITE_GC_URL && window.SITE_GC_URL.indexOf('YOUR-SITE') === -1;
  }

  /* Fire a page_view once the count() helper is available. */
  window.addEventListener('load', function () {
    if (!isConfigured()) return;
    if (!window.goatcounter || typeof window.goatcounter.count !== 'function') return;
    window.goatcounter.count({
      path: location.pathname + location.search + location.hash
    });
  });

  /* Public helper used by course-page.js to record module_open events. */
  window.trackModuleOpen = function (subjectSlug, moduleNum) {
    if (!isConfigured()) return;
    if (!window.goatcounter || typeof window.goatcounter.count !== 'function') return;
    try {
      window.goatcounter.count({
        path:  'module_open/' + subjectSlug + '/' + moduleNum,
        title: 'Module open: ' + subjectSlug + ' #' + moduleNum,
        event: true
      });
    } catch (e) { /* ignore */ }
  };
})();
