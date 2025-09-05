# run_scraper.py
import logging
# Import the functions you need from your main app file
from app import fetch_and_store_articles, init_db

if __name__ == '__main__':
    # Set up basic logging for the cron job
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Cron job started: Initializing DB schema.")
    # Ensure the database tables exist before running the scraper
    init_db()

    logging.info("Starting the article fetching process...")
    # Call the main scraper function
    fetch_and_store_articles()
    logging.info("Cron job finished.")