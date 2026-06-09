# screenshot_map.py
import os
import sys
from playwright.sync_api import sync_playwright

def screenshot_html(html_path, out_png, width=1400, height=900, wait_ms=2000):
    # Convert to absolute path
    html_abspath = os.path.abspath(html_path)

    # FIX: Convert backslashes BEFORE using in f-string
    safe_path = html_abspath.replace("\\", "/")

    # Construct proper file:// URL
    url = f"file:///{safe_path}"

    print("Loading:", url)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(url)
        page.wait_for_timeout(wait_ms)  # wait for map tiles to load
        page.screenshot(path=out_png, full_page=False)
        browser.close()

    print("Screenshot saved:", out_png)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python screenshot_map.py input.html output.png")
        sys.exit(1)

    html_file = sys.argv[1]
    out_file = sys.argv[2]

    screenshot_html(html_file, out_file)
