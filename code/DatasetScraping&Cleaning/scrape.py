import pandas as pd
from io import StringIO
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# Path to your ChromeDriver
chrome_driver_path = "E:\\chromedriver-win64\\chromedriver.exe"

# Setup Chrome options
options = Options()
options.add_argument("--headless")  # Run in headless mode (optional)

# Initialize WebDriver
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=options)

# URL of the webpage to scrape
url = "https://github.com/MayCooper/Product-Price-Prediction/blob/main/Updated%20Project/Car%20details%20v3.xls"
driver.get(url)

# Wait for the element to load (if necessary)
driver.implicitly_wait(10)

# Locate the <textarea> element using an appropriate selector
textarea = driver.find_element(By.ID, "read-only-cursor-text-area")

# Extract and print the text content
car_text = textarea.get_attribute("value")  # Use "value" for <textarea>
print(car_text)

# Close the browser
driver.quit()

df = pd.read_csv(StringIO(car_text))
df.to_csv('scraped_dataset.csv', index=False)
print(df)
