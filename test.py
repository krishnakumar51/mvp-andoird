import time
from pathlib import Path
from appium import webdriver
from appium.options.android import UiAutomator2Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- Configuration ---
APPIUM_SERVER_URL = "http://localhost:4723"
TARGET_URL = "https://in.iherb.com/"
SEARCH_TERM = "best face masks"
CAPTURE_DIR = Path("html_captures")

def run_search_test():
    """
    A standalone test script to verify the two-step search process on iHerb,
    capturing HTML at each critical step for debugging.
    """
    driver = None
    print("Initializing Appium driver for the test...")
    try:
        # Create the capture directory if it doesn't exist
        CAPTURE_DIR.mkdir(exist_ok=True)
        print(f"HTML captures will be saved in: {CAPTURE_DIR.resolve()}")

        # --- Driver Setup ---
        options = UiAutomator2Options()
        options.platform_name = 'Android'
        options.udid = "ZD222GXYPV"
        options.automation_name = 'UiAutomator2'
        options.browser_name = "Chrome"
        options.no_reset = True
        options.auto_grant_permissions = True
        options.set_capability("appium:chromedriver_autodownload", True)
        options.set_capability("goog:chromeOptions", {"w3c": True, "args": ["--disable-fre", "--no-first-run"]})

        driver = webdriver.Remote(APPIUM_SERVER_URL, options=options)
        wait = WebDriverWait(driver, 20)
        
        print(f"Navigating to {TARGET_URL}...")
        driver.get(TARGET_URL)
        time.sleep(7)

        # --- STEP 1: Capture initial HTML ---
        print("Capturing initial page source...")
        with open(CAPTURE_DIR / "01_before_tap.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        print("Saved '01_before_tap.html'.")

        # --- STEP 2: Find and tap the placeholder search bar ---
        print("Looking for the search placeholder...")
        search_placeholder_xpath = "//div[text()='Search']"
        search_placeholder = wait.until(
            EC.element_to_be_clickable((By.XPATH, search_placeholder_xpath))
        )
        print("Search placeholder found. Tapping it...")
        search_placeholder.click()
        
        time.sleep(5)

        # --- STEP 3: Capture HTML *after* the tap ---
        print("Capturing page source after tapping placeholder...")
        with open(CAPTURE_DIR / "02_after_tap.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        print("Saved '02_after_tap.html'.")

        # --- STEP 4: Find the REAL input, fill it, and submit ---
        print("Looking for the actual search input field...")
        # CORRECTED: This selector is based on the analysis of 02_after_tap.html
        search_input_selector = "input#txtSearch" 
        search_input = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, search_input_selector))
        )
        print(f"Search input found. Typing '{SEARCH_TERM}' and submitting...")
        search_input.send_keys(SEARCH_TERM)
        search_input.send_keys(Keys.ENTER)

        # --- VERIFICATION ---
        wait.until(EC.url_contains("search?kw="))
        print("\n--- TEST SUCCESSFUL ---")
        print(f"Successfully navigated to the search results page for '{SEARCH_TERM}'.")
        
        # --- FINAL CAPTURE ---
        print("Capturing final search results page source...")
        with open(CAPTURE_DIR / "03_after_search.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        print("Saved '03_after_search.html'.")


    except Exception as e:
        print(f"\n--- AN ERROR OCCURRED DURING THE TEST ---")
        print(f"Error: {e}")
        if driver:
            error_screenshot = Path("error_screenshot.png")
            driver.save_screenshot(str(error_screenshot))
            print(f"Saved a screenshot of the error state to: {error_screenshot.resolve()}")
        
    finally:
        if driver:
            print("Closing driver session.")
            driver.quit()

if __name__ == "__main__":
    run_search_test()

