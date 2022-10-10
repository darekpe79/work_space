from selenium import webdriver
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
driver = webdriver.Chrome(r"F:\chromedriver.exe")
driver.get("https://www.nytimes.com")
headlines = driver.find_elements(By.CLASS_NAME, 'css-xdandi') 
for headline in headlines:
    print(headline.text.strip())
    

driver = webdriver.Chrome(r"F:\chromedriver.exe")
driver.get("https://kansalliskirjasto.finna.fi/Search/Results?limit=0&hiddenFilters%5B%5D=%23%3A%22%28building%3A0%2FJOURNALFI%2F%29+OR+%28building%3A1%2FNLF%2Farto%2F%29%22&type=AllFields&filter%5B%5D=~building%3A%220%2FJOURNALFI%2F%22")
headlines = driver.find_elements(By.CLASS_NAME, "title-container") 
for headline in headlines:
    print(headline.text.strip())
    
click_name=headlines = driver.find_element(By.LINK_TEXT, "/Search/Results?filter%5B%5D=%7Ebuilding%3A%220%2FJOURNALFI%2F%22&amp;hiddenFilters%5B%5D=%23%3A%22%28building%3A0%2FJOURNALFI%2F%29+OR+%28building%3A1%2FNLF%2Farto%2F%29%22&amp;type=AllFields&amp;page=2")