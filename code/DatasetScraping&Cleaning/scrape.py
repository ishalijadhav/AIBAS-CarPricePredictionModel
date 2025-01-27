import pandas as pd
from io import StringIO
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# Path to your ChromeDriver
chrome_driver_path = "E:\\chromedriver-win64\\chromedriver.exe"

options = Options()
options.add_argument("--headless") 

# Initialize WebDriver
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=options)

# URL of the webpage to scrape
url = "https://github.com/MayCooper/Product-Price-Prediction/blob/main/Updated%20Project/Car%20details%20v3.xls"
driver.get(url)
driver.implicitly_wait(10)

# Locating the <textarea> element
textarea = driver.find_element(By.ID, "read-only-cursor-text-area")

car_text = textarea.get_attribute("value") 
print(car_text)

driver.quit()

df = pd.read_csv(StringIO(car_text))
df.to_csv('scraped_dataset.csv', index=False)
print(df)
