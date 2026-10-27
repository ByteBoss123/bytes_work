"""Start the scoring service, replay real formulary rows through all alert phases, and screenshot
/dashboard in headless Chromium (desktop light, desktop dark, phone width).

Usage (needs `pip install playwright && python -m playwright install --with-deps chromium`):
    PYTHONPATH=. python tests/dashboard_screenshot.py
Writes reports/dashboard_desktop.png, reports/dashboard_dark.png, reports/dashboard_mobile.png
and reports/dashboard_screenshot.json (console errors and the values shown on the cards).
"""
import json
import tempfile
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from pbm.service import ScoringService, make_server


def traffic(s):
    pairs = json.loads(Path('reports/reference_pairs.json').read_text())
    prior_pa = [r for r in pairs if r['prev_pair_pa'] == 1][:50]
    for i in range(20):  # normal
        s.record(200, 0, s.predict({'pairs': pairs[i * 50:(i + 1) * 50]}))
    for _ in range(6):  # invalid payloads -> failure-rate alert
        s.record(400, 0, [])
    for _ in range(20):  # real subpopulation shift -> drift alert
        s.record(200, 0, s.predict({'pairs': prior_pa}))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        s = ScoringService('reports/model.joblib', 'reports/monitoring_reference.json', Path(tmp) / 't.sqlite')
        traffic(s)
        server = make_server(s, 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f'http://127.0.0.1:{server.server_port}/dashboard'
        out, errors = {}, []
        with sync_playwright() as p:
            browser = p.chromium.launch()
            for name, kw in [('desktop', {'viewport': {'width': 1200, 'height': 900}}),
                             ('dark', {'viewport': {'width': 1200, 'height': 900}, 'color_scheme': 'dark'}),
                             ('mobile', {'viewport': {'width': 390, 'height': 844}, 'device_scale_factor': 2})]:
                page = browser.new_page(**kw)
                page.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
                page.on('pageerror', lambda e: errors.append(str(e)))
                page.goto(url)
                page.wait_for_selector('#cards .value')
                time.sleep(1)
                page.screenshot(path=f'reports/dashboard_{name}.png', full_page=True)
                out[name] = {'cards': page.eval_on_selector_all('#cards .card', 'els => els.map(e => e.innerText)'),
                             'horizontal_overflow': page.evaluate('document.documentElement.scrollWidth > window.innerWidth')}
                page.close()
            browser.close()
        server.shutdown()
        server.server_close()
    result = {'console_errors': errors, 'views': out, 'passed': not errors and not any(v['horizontal_overflow'] for v in out.values())}
    Path('reports/dashboard_screenshot.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
