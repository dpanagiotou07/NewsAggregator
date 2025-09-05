# app.py
# Description: A single-file Flask application for news aggregation, AI processing, and semantic search.
# This file combines the logic from the original config, database, ai_services, and scraper modules.
# MODIFIED: Implemented a new, advanced summarization prompt for higher quality summaries.
# MODIFIED: Added news categorization using Gemini.
# MODIFIED: Consolidated summary display in the frontend for a cleaner look.
# MODIFIED: Improved title extraction for conciseness using more aggressive HTML parsing.
# MODIFIED: Implemented dynamic topic creation and assignment using LLM and vector similarity.
# MODIFIED: Added multilingual support (English, Greek, Russian) for summaries and static texts.
# MODIFIED: Implemented language selection via flags and cookie storage for persistence.
# MODIFIED: Added on-the-fly translation fallback for missing summaries.
# MODIFIED: Implemented a strict AI-first image policy for political articles to avoid copyright issues.
# NEW: Implemented aggressive markdown cleaning before AI processing to improve accuracy.
# NEW: Frontend rendering is now robust against processing failures, displaying clean messages instead of raw data.
# NEW: Implemented a more reliable translation pipeline for summaries.
# FIXED: The `find_or_create_topic` function's similarity threshold has been lowered to group articles more effectively.
# FIXED: The frontend JavaScript has been updated to prevent the translation bug.
# NEW: Made the main title in the header a clickable link to the homepage.

# --- IMPORTS (Consolidated from all modules) ---
import logging
import os
import threading
import time
import asyncio
import re
import base64
import smtplib
import ssl
import json
from email.message import EmailMessage
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from types import SimpleNamespace
from collections import defaultdict

# --- Third-party Libraries ---
import psycopg
import requests
import pytz
import trafilatura
import google.auth
import google.auth.transport.requests
from google.oauth2 import service_account # ADDED FOR ENV VAR AUTH
from flask import Flask, jsonify, render_template_string, request, make_response # Modified for cookies
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from newspaper import Article
from crawl4ai import AsyncWebCrawler
from psycopg.rows import dict_row
from bs4 import BeautifulSoup
from numpy import dot
from numpy.linalg import norm
from dotenv import load_dotenv

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

# --- LOGGING CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# --- SECTION 1: CONFIGURATION (from config.py) ---
# ==============================================================================

# --- Database Configuration ---
# Render provides a single URL for the database
DATABASE_URL = os.environ.get('DATABASE_URL')

# --- AI & Cloud Configuration ---
GOOGLE_CLOUD_PROJECT_ID = "news-aggregator-465312"

# API Endpoints for Vertex AI
GEMINI_API_URL = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{GOOGLE_CLOUD_PROJECT_ID}/locations/us-central1/publishers/google/models/gemini-2.0-flash-001:generateContent"
IMAGEN_API_URL = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{GOOGLE_CLOUD_PROJECT_ID}/locations/us-central1/publishers/google/models/imagen-3.0-generate-002:predict"

# --- Scraper Configuration ---
HOMEPAGE_SITES_TO_SCRAPE = [
    'https://politis.com.cy/',
    'https://www.cna.org.cy/',
    'https://www.sigmalive.com/',
    'https://cyprus-mail.com/'
]
MAX_RETRIES = 3
INITIAL_BACKOFF = 5 # seconds

EXCLUDE_URL_PATTERNS = [
    "*login*", "*subscribe*", "*contact*", "*about*", "*privacy*",
    "*terms*", "*advertise*", "*profile*", "*search*", "*cookie*", "*popup*",
    "*category*", "*section*", "*topic*", "*review*"
]

EXCLUDE_KEYWORD = 'north cyprus'
CYPRUS_TIMEZONE = 'Europe/Nicosia'

# --- Email Sending Configuration ---
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 465
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD")


# --- Model Configuration ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
# This threshold is for comparing one article summary to another.
# 0.80 is a strong score that ensures topics are highly related.
COSINE_SIMILARITY_THRESHOLD = 0.80

# --- News Categories ---
NEWS_CATEGORIES = [
    "Politics", "Sports", "Technology", "Economy", "Health",
    "Environment", "Culture", "World News", "Local News", "Education",
    "Crime", "Science", "Business", "Lifestyle", "Entertainment"
]

# --- Localization Configuration ---
LANGUAGES = {
    'en': 'English',
    'el': 'Ελληνικά',
    'ru': 'Русский'
}
DEFAULT_LANG = 'en'

# Static texts localization
LOCALIZED_TEXTS = {
    'en': {
        'subscribe_button': 'Subscribe',
        'subscribe_digest': 'Subscribe to the Digest',
        'enter_email_updates': 'Enter your email to receive morning and evening updates.',
        'subscribe_now': 'Subscribe Now',
        'search_placeholder': 'Search by keyword...',
        'browse_by_category': 'Browse by Category:',
        'browse_by_topic': 'Browse by Topic:',
        'back_to_all_news': '← Back to All News',
        'todays_news': "Today's News",
        'previous_articles': 'Previous Articles',
        'no_new_articles_today': 'No new articles today. Check back later!',
        'no_previous_articles': 'No previous articles found.',
        'article_not_available': 'Article not available',
        'summary_not_available': 'Summary not available.',
        'image_not_available': 'Image not available',
        'find_similar': 'Find Similar',
        'cyprus_news_digest': 'Cyprus News Digest',
        'ai_summaries_search': 'AI Summaries & Semantic Search',
        'footer_text': '&copy; 2025 Cyprus News Digest. Built with Flask, PostgreSQL & Google AI.',
        'network_error': 'Failed to load content.',
        'search_results': 'Keyword Search: "{query}"',
        'no_articles_search': 'No articles found for your search.',
        'similar_articles_title': 'Showing Semantically Similar Articles',
        'no_similar_articles': 'Could not find any similar articles.',
        'category_title': 'Category: "{category}"',
        'no_articles_category': 'No articles found in the "{category}" category.',
        'topic_title': 'Topic: "{topic}"',
        'no_articles_topic': 'No articles found for topic "{topic}".',
        'subscribing': 'Subscribing...',
        'subscription_success': 'Thank you for subscribing! A welcome email has been sent.',
        'subscription_failed': 'Subscription failed.',
        'invalid_email': 'Invalid email address.',
        'already_subscribed': 'This email is already subscribed.',
        'unexpected_error': 'An unexpected error occurred.',
        'summary_not_available_note': '(Translated from English)'
    },
    'el': {
        'subscribe_button': 'Εγγραφή',
        'subscribe_digest': 'Εγγραφείτε στο Digest',
        'enter_email_updates': 'Εισάγετε το email σας για πρωινές και βραδινές ενημερώσεις.',
        'subscribe_now': 'Εγγραφείτε τώρα',
        'search_placeholder': 'Αναζήτηση με λέξη-κλειδί...',
        'browse_by_category': 'Περιήγηση ανά Κατηγορία:',
        'browse_by_topic': 'Περιήγηση ανά Θέμα:',
        'back_to_all_news': '← Πίσω σε Όλες τις Ειδήσεις',
        'todays_news': 'Ειδήσεις Σήμερα',
        'previous_articles': 'Προηγούμενα Άρθρα',
        'no_new_articles_today': 'Δεν υπάρχουν νέα άρθρα σήμερα. Ελέγξτε αργότερα!',
        'no_previous_articles': 'Δεν βρέθηκαν προηγούμενα άρθρα.',
        'article_not_available': 'Άρθρο μη διαθέσιμο',
        'summary_not_available': 'Περίληψη μη διαθέσιμη.',
        'image_not_available': 'Εικόνα μη διαθέσιμη',
        'find_similar': 'Βρες Παρόμοια',
        'cyprus_news_digest': 'Cyprus News Digest',
        'ai_summaries_search': 'Περιλήψεις AI & Σημασιολογική Αναζήτηση',
        'footer_text': '&copy; 2025 Cyprus News Digest. Κατασκευάστηκε με Flask, PostgreSQL & Google AI.',
        'network_error': 'Αποτυχία φόρτωσης περιεχομένου.',
        'search_results': 'Αναζήτηση λέξεων-κλειδιών: "{query}"',
        'no_articles_search': 'Δεν βρέθηκαν άρθρα για την αναζήτησή σας.',
        'similar_articles_title': 'Εμφάνιση Σημασιολογικά Παρόμοιων Άρθρων',
        'no_similar_articles': 'Δεν βρέθηκαν παρόμοια άρθρα.',
        'category_title': 'Κατηγορία: "{category}"',
        'no_articles_category': 'Δεν βρέθηκαν άρθρα στην κατηγορία "{category}".',
        'topic_title': 'Θέμα: "{topic}"',
        'no_articles_topic': 'Δεν βρέθηκαν άρθρα για το θέμα "{topic}".',
        'subscribing': 'Εγγραφή...',
        'subscription_success': 'Ευχαριστούμε για την εγγραφή σας! Ένα email καλωσορίσματος έχει σταλεί.',
        'subscription_failed': 'Η εγγραφή απέτυχε.',
        'invalid_email': 'Μη έγκυρη διεύθυνση email.',
        'already_subscribed': 'Αυτό το email είναι ήδη εγγεγραμμένο.',
        'unexpected_error': 'Προέκυψε ένα απρόσμενο σφάλμα.',
        'summary_not_available_note': '(Μεταφρασμένο από τα Αγγλικά)'
    },
    'ru': {
        'subscribe_button': 'Подписаться',
        'subscribe_digest': 'Подпишитесь на Дайджест',
        'enter_email_updates': 'Введите ваш email, чтобы получать утренние и вечерние обновления.',
        'subscribe_now': 'Подписаться сейчас',
        'search_placeholder': 'Поиск по ключевому слову...',
        'browse_by_category': 'Просмотр по Категориям:',
        'browse_by_topic': 'Просмотр по Темам:',
        'back_to_all_news': '← Назад ко Всем Новостям',
        'todays_news': 'Новости Сегодня',
        'previous_articles': 'Предыдущие Статьи',
        'no_new_articles_today': 'Новых статей сегодня нет. Загляните позже!',
        'no_previous_articles': 'Предыдущие статьи не найдены.',
        'article_not_available': 'Статья недоступна',
        'summary_not_available': 'Сводка недоступна.',
        'image_not_available': 'Изображение недоступно',
        'find_similar': 'Найти Похожие',
        'cyprus_news_digest': 'Cyprus News Digest',
        'ai_summaries_search': 'AI Сводки и Семантический Поиск',
        'footer_text': '&copy; 2025 Cyprus News Digest. Создано с помощью Flask, PostgreSQL и Google AI.',
        'network_error': 'Не удалось загрузить контент.',
        'search_results': 'Поиск по ключевому слову: "{query}"',
        'no_articles_search': 'Статей по вашему запросу не найдено.',
        'similar_articles_title': 'Показ Семантически Похожих Статей',
        'no_similar_articles': 'Похожих статей не найдено.',
        'category_title': 'Категория: "{category}"',
        'no_articles_category': 'Статей в категории "{category}" не найдено.',
        'topic_title': 'Тема: "{topic}"',
        'no_articles_topic': 'Статей по теме "{topic}" не найдено.',
        'subscribing': 'Подписка...',
        'subscription_success': 'Спасибо за подписку! Приветственное письмо было отправлено.',
        'subscription_failed': 'Подписка не удалась.',
        'invalid_email': 'Неверный адрес электронной почты.',
        'already_subscribed': 'Этот адрес электронной почты уже подписан.',
        'unexpected_error': 'Произошла непредвиденная ошибка.',
        'summary_not_available_note': '(Переведено с английского)'
    }
}

# Localized categories
LOCALIZED_CATEGORIES = {
    'en': NEWS_CATEGORIES,
    'el': [
        "Πολιτική", "Αθλητισμός", "Τεχνολογία", "Οικονομία", "Υγεία",
        "Περιβάλλον", "Πολιτισμός", "Παγκόσμια Νέα", "Τοπικά Νέα", "Εκπαίδευση",
        "Έγκλημα", "Επιστήμη", "Επιχειρήσεις", "Τρόπος Ζωής", "Ψυχαγωγία"
    ],
    'ru': [
        "Политика", "Спорт", "Технологии", "Экономика", "Здоровье",
        "Окружающая среда", "Культура", "Мировые Новости", "Местные Новости", "Образование",
        "Преступность", "Наука", "Бизнес", "Образ жизни", "Развлечения"
    ]
}


# ==============================================================================
# --- SECTION 2: AI SERVICES (from ai_services.py) ---
# ==============================================================================

# --- Sentence Transformer Model Initialization ---
logging.info("Loading sentence transformer model...")
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logging.info("Sentence transformer model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load sentence transformer model: {e}")
    embedding_model = None

# --- Placeholder Images (Base64 encoded strings) ---
POLITICAL_PLACEHOLDER_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=" 
GENERIC_PLACEHOLDER_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="


def get_political_placeholder():
    return POLITICAL_PLACEHOLDER_BASE64

def get_generic_placeholder():
    return GENERIC_PLACEHOLDER_BASE64


# --- AI Helper Functions ---

def cosine_similarity(vec1, vec2):
    """Calculates the cosine similarity between two vectors."""
    if vec1 is None or vec2 is None:
        return -1 # Indicate no similarity
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def make_api_request(url, payload, headers, timeout=120):
    """Makes a POST request to a given API endpoint."""
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error for {url}: {http_err} - {http_err.response.text}")
    except requests.exceptions.RequestException as e:
        logging.error(f"API request to {url} failed: {e}")
    return None

# --- REPLACED AUTHENTICATION FUNCTION ---
def get_google_ai_headers():
    """Generates OAuth2 headers by loading credentials from an environment variable."""
    try:
        creds_info = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if not creds_info:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable not set.")
        
        creds_dict = json.loads(creds_info)
        creds = service_account.Credentials.from_service_account_info(
            creds_dict,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        
        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req)
        
        return {"Authorization": f"Bearer {creds.token}", "Content-Type": "application/json"}
        
    except Exception as e:
        logging.error(f"CRITICAL: Could not get Google Auth credentials: {e}", exc_info=True)
        return None

# --- Core AI Functions ---

# New: Multilingual summarization prompt templates
SUMMARY_PROMPTS = {
    'en': """
Your task is to summarize a news article for a Cypriot media outlet. The goal is to create a brief, factual, and clear summary that allows a reader to quickly understand the main points of the news.
Follow these guidelines:
1. Identify the Core News: Focus on the single most important piece of information in the article. What is the key event, announcement, or finding? Your summary should lead with this.
2. Be Concise and Factual: Write a single, dense paragraph. Report the essential facts without adding opinions or interpretations.
3. Include Key Information (The 5 Ws): Ensure your summary answers the fundamental questions: Who, What, Where, When, Why/How.
4. What to Exclude: Omit direct quotes, minor details, lengthy background.
The final summary should be a stand-alone piece of text that gives a complete, albeit high-level, overview of the original article.

**IMPORTANT: The summary MUST be written in English, regardless of the article's original language.**

Article Text:
'''
{text_content}
'''
""",
    'el': """
Ο στόχος σας είναι να συνοψίσετε ένα ειδησεογραφικό άρθρο για ένα κυπριακό μέσο ενημέρωσης. Ο σκοπός είναι να δημιουργήσετε μια σύντομη, πραγματική και σαφή περίληψη που θα επιτρέπει στον αναγνώστη να κατανοήσει γρήγορα τα κύρια σημεία των ειδήσεων.
Ακολουθήστε αυτές τις οδηγίες:
1. Εντοπίστε τον Πυρήνα των Ειδήσεων: Επικεντρωθείτε στην πιο σημαντική πληροφορία του άρθρου. Ποιο είναι το βασικό γεγονός, η ανακοίνωση ή το εύρημα; Η περίληψή σας πρέπει να ξεκινάει από αυτό.
2. Να είστε Συνοπτικοί και Πραγματικοί: Γράψτε μια ενιαία, περιεκτική παράγραφο. Αναφέρετε τα ουσιώδη γεγονότα χωρίς να προσθέτετε απόψεις ή ερμηνείες.
3. Συμπεριλάβετε Βασικές Πληροφορίες (τα 5 W): Βεβαιωθείτε ότι η περίληψή σας απαντά στις βασικές ερωτήσεις: Ποιος, Τι, Πού, Πότε, Γιατί/Πώς.
4. Τι να Εξαιρέσετε: Παραλείψτε τα άμεσα αποσπάσματα, τις δευτερεύουσες λεπτομέρειες, το εκτεταμένο ιστορικό.
Η τελική περίληψη πρέπει να είναι ένα αυτόνομο κείμενο που δίνει μια ολοκληρωμένη, αν και υψηλού επιπέδου, επισκόπηση του αρχικού άρθρου.

Κείμενο Άρθρου:
'''
{text_content}
'''
""",
    'ru': """
Ваша задача — кратко изложить новостную статью для кипрского СМИ. Цель состоит в том, чтобы создать краткое, фактическое и ясное резюме, которое позволит читателю быстро понять основные моменты новости.
Следуйте этим рекомендациям:
1. Определите Основную Новость: Сосредоточьтесь на самой важной информации в статье. Каково ключевое событие, объявление или вывод? Ваше резюме должно начинаться с этого.
2. Будьте Лаконичны и Фактичны: Напишите один плотный абзац. Сообщите существенные факты, не добавляя мнений или интерпретаций.
3. Включите Ключевую Информацию (5 W): Убедитесь, что ваше резюме отвечает на основные вопросы: Кто, Что, Где, Когда, Почему/Как.
4. Что Исключить: Опустите прямые цитаты, второстепенные детали, обширные предыстории.
Окончательное резюме должно быть самостоятельным текстом, который дает полное, хотя и общее, представление об исходной статье.

Текст статьи:
'''
{text_content}
'''
"""
}

def summarize_with_gemini(text_content, lang='en', is_short=False):
    """
    Generates a summary for the given text using a specific prompt for the target language.
    Added `is_short` to differentiate between long and short summary prompts.
    """
    if not text_content or len(text_content.split()) < 30:
        return None

    # Determine which prompt template to use based on language and length requirement
    if is_short:
        prompt_template = "Generate a single, compelling sentence that captures the main point of the following article in {lang_name}: {text_content}"
        # Adjust language name for prompt
        lang_name = LANGUAGES.get(lang, 'English') # Default to English name if not found
        prompt = prompt_template.format(text_content=text_content, lang_name=lang_name)
    else:
        prompt_template = SUMMARY_PROMPTS.get(lang, SUMMARY_PROMPTS['en']) # Default to English if lang not found
        prompt = prompt_template.format(text_content=text_content)

    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }]
    }
    headers = get_google_ai_headers()
    if not headers: return None

    try:
        data = make_api_request(GEMINI_API_URL, payload, headers)
        if data and 'candidates' in data and data['candidates'][0]['content']['parts']:
            return data['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        logging.error(f"Gemini summarization failed for lang '{lang}': {e}")
    return None

def translate_with_gemini(text_content, target_lang):
    """
    Translates text content using Gemini.
    """
    if not text_content or not target_lang:
        return None
   
    target_lang_name = LANGUAGES.get(target_lang, 'English')
   
    prompt = f"""
    Translate the following news summary into {target_lang_name}.
    Summary to translate:
    '''
    {text_content}
    '''
    """

    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }]
    }
    headers = get_google_ai_headers()
    if not headers: return None
   
    try:
        data = make_api_request(GEMINI_API_URL, payload, headers)
        if data and 'candidates' in data and data['candidates'][0]['content']['parts']:
            return data['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        logging.error(f"Gemini translation failed for lang '{target_lang}': {e}")
    return None

def categorize_with_gemini(text_content, categories):
    """Categorizes the given text into one of the predefined categories using Gemini."""
    if not text_content:
        return "Uncategorized"

    # Use English categories for categorization to keep it consistent for the model
    category_list_str = ", ".join(NEWS_CATEGORIES)
    prompt = f"""
    Analyze the following news article and classify it into one single category from the following list: {category_list_str}.
    If the article does not clearly fit into any of these categories, classify it as "Uncategorized".
    Return ONLY the category name, without any additional text or punctuation.

    Article Text:
    '''
    {text_content}
    '''
    Category:
    """

    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }]
    }
    headers = get_google_ai_headers()
    if not headers:
        return "Uncategorized"

    try:
        data = make_api_request(GEMINI_API_URL, payload, headers, timeout=60)
        if data and 'candidates' in data and data['candidates'][0]['content']['parts']:
            category = data['candidates'][0]['content']['parts'][0]['text'].strip()
            # Basic validation to ensure the response is one of the predefined categories
            if category in NEWS_CATEGORIES:
                return category
            elif "Uncategorized" in category: # Gemini might add more text, check if 'Uncategorized' is present
                return "Uncategorized"
            else:
                logging.warning(f"Gemini returned an unlisted category: '{category}'. Defaulting to Uncategorized.")
                return "Uncategorized"
    except Exception as e:
        logging.error(f"Gemini categorization failed: {e}")
    return "Uncategorized"

def create_topic_name_with_gemini(article_title: str, article_summary: str) -> str:
    """
    Asks LLM to create a concise, 3-word topic title based on the article's title and summary.
    This will remain in English for consistency in topics.
    """
    if not article_title and not article_summary:
        return "General Topic"

    text_for_llm = f"Title: {article_title}\nSummary: {article_summary}"
    
    prompt = f"""
    You are an expert news editor. Based on the article title and summary, create a short, general topic tag.
    RULES:
    1.  **Be General:** The topic must be broad enough to group multiple related articles. Think of it as a news "saga" or a recurring issue.
    2.  **Length:** Use exactly two (2) words. For example: "Paphos Wildfires", "Ukraine Conflict", "Government Policy", "Economic Concerns".
    3.  **Avoid Specifics:** Do NOT use names of specific people, villages, or dates unless they define a major, long-running event.
    4.  **Example:** An article titled "Firefighters battle blaze near Chlorakas" should have the topic "Paphos Wildfires", NOT "Chlorakas Fire". An article about Trump's comments on Ukraine should be "Ukraine Conflict", NOT "Trump Ukraine Remarks".

    Return ONLY the two-word topic.

    Article Content:
    '''
    {text_for_llm}
    '''
    Two-Word Topic:
    """

    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }]
    }
    headers = get_google_ai_headers()
    if not headers:
        return "General Topic"

    try:
        data = make_api_request(GEMINI_API_URL, payload, headers, timeout=30)
        if data and 'candidates' in data and data['candidates'][0]['content']['parts']:
            topic_raw = data['candidates'][0]['content']['parts'][0]['text'].strip()
            # Clean and ensure it's roughly 2 words
            topic_words = re.findall(r'\b\w+\b', topic_raw)
            if len(topic_words) >= 1 and len(topic_words) <= 3: # Allow slight flexibility
                return " ".join(topic_words).title() # Capitalize each word
            else:
                logging.warning(f"Gemini returned a non-2-word topic '{topic_raw}'. Using fallback.")
                return "General News"
    except Exception as e:
        logging.error(f"Gemini topic creation failed: {e}")
    return "General News"


def generate_image_with_imagen(prompt_text):
    """Generates an image using Imagen 3.0 based on a text prompt."""
    if not prompt_text:
        return None

    headers = get_google_ai_headers()
    if not headers:
        return None

    image_prompt = f"A photorealistic, high-quality news-style image inspired by: {prompt_text}. Cinematic lighting, no text, no logos."
    payload = {"instances": [{"prompt": image_prompt}], "parameters": {"sampleCount": 1}}
    data = make_api_request(IMAGEN_API_URL, payload, headers)
    try:
        if data and 'predictions' in data and data['predictions']:
            return data['predictions'][0].get('bytesBase64Encoded')
    except (KeyError, IndexError, TypeError) as e:
        logging.error(f"Could not parse Imagen response: {e}. Response: {data}")
    return None

def restyle_image_with_imagen(base64_data):
    """Restyles a given base64 encoded image using an AI prompt."""
    if not base64_data: return None
    headers = get_google_ai_headers()
    if not headers: return None

    prompt = "A vibrant, cartoon-style illustration based on the provided image. Maintain the core subject and composition, but reinterpret it with a clean, graphic, illustrated look. Do not include any text."
    payload = {
        "instances": [{
            "prompt": prompt,
            "image": {"bytesBase64Encoded": base64_data}
        }],
        "parameters": {"sampleCount": 1}
    }
    data = make_api_request(IMAGEN_API_URL, payload, headers)
    if data and 'predictions' in data and data['predictions']:
        return data['predictions'][0].get('bytesBase64Encoded')
    return None

def download_and_encode_image(image_url):
    """Downloads an image from a URL and returns it as a base64 string."""
    if not image_url or not image_url.startswith(('http:', 'https:')): return None
    try:
        response = requests.get(image_url, timeout=15, stream=True, allow_redirects=True, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        if 'image' not in response.headers.get('Content-Type', ''): return None
        return base64.b64encode(response.content).decode('utf-8')
    except Exception as e:
        logging.error(f"Failed to download image from {image_url}: {e}")
        return None

def get_embedding(text):
    """Generates a vector embedding for a given text."""
    if not text or not embedding_model:
        return None
    try:
        return embedding_model.encode(text)
    except Exception as e:
        logging.error(f"Failed to generate embedding: {e}")
        return None

# ==============================================================================
# --- SECTION 3: DATABASE (from database.py) ---
# ==============================================================================

def get_db_connection():
    """Establishes a connection to the PostgreSQL database using a connection URL."""
    try:
        # This will read the URL from Render's environment or your local .env file
        conn_string = os.environ.get('DATABASE_URL')
        if not conn_string:
            raise ValueError("DATABASE_URL environment variable is not set.")
        
        conn = psycopg.connect(conn_string, row_factory=dict_row)
        return conn
    except psycopg.OperationalError as e:
        logging.error(f"CRITICAL: Could not connect to PostgreSQL database: {e}")
        return None

def init_db():
    """
    Initializes the database schema, creating tables for articles, subscribers, and topics.
    """
    conn = get_db_connection()
    if not conn:
        logging.error("Database connection failed, cannot initialize.")
        return

    try:
        with conn.cursor() as cur:
            logging.info("Enabling pgvector extension...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            logging.info("Creating 'topics' table...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS topics (
                    id SERIAL PRIMARY KEY,
                    topic_name TEXT NOT NULL UNIQUE,
                    topic_embedding VECTOR(384) NOT NULL,
                    article_count INTEGER DEFAULT 1,
                    is_confirmed BOOLEAN DEFAULT FALSE
                );
            """)

            logging.info("Creating 'articles' table with vector support, 'category' and 'topic_id' columns...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL UNIQUE,
                    summary_en TEXT,
                    summary_el TEXT,
                    summary_ru TEXT,
                    short_summary TEXT,
                    source TEXT,
                    image_base64 TEXT,
                    published_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    summary_embedding VECTOR(384),
                    category TEXT DEFAULT 'Uncategorized',
                    topic_id INTEGER REFERENCES topics(id)
                );
            """)

            logging.info("Creating 'subscribers' table...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS subscribers (
                    id SERIAL PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    subscribed_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # ### FIX 1: Add the new columns to the topics table if they don't exist ###
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='topics' AND column_name='article_count') THEN
                        ALTER TABLE topics ADD COLUMN article_count INTEGER DEFAULT 1;
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='topics' AND column_name='is_confirmed') THEN
                        ALTER TABLE topics ADD COLUMN is_confirmed BOOLEAN DEFAULT FALSE;
                    END IF;
                END
                $$;
            """)

            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='articles' AND column_name='category') THEN
                        ALTER TABLE articles ADD COLUMN category TEXT DEFAULT 'Uncategorized';
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='articles' AND column_name='topic_id') THEN
                        ALTER TABLE articles ADD COLUMN topic_id INTEGER;
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='articles' AND column_name='summary_en') THEN
                        ALTER TABLE articles ADD COLUMN summary_en TEXT;
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='articles' AND column_name='summary_el') THEN
                        ALTER TABLE articles ADD COLUMN summary_el TEXT;
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='articles' AND column_name='summary_ru') THEN
                        ALTER TABLE articles ADD COLUMN summary_ru TEXT;
                    END IF;
                    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='articles' AND column_name='summary') THEN
                        UPDATE articles SET summary_en = summary WHERE summary_en IS NULL;
                        ALTER TABLE articles DROP COLUMN summary;
                    END IF;
                END
                $$;
            """)

            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'articles_topic_id_fkey') THEN
                        ALTER TABLE articles ADD CONSTRAINT articles_topic_id_fkey FOREIGN KEY (topic_id) REFERENCES topics(id);
                    END IF;
                END
                $$;
            """)

            logging.info("Creating IVFFlat index on vector column for faster similarity search on articles...")
            cur.execute("""
                CREATE INDEX IF NOT EXISTS articles_embedding_idx
                ON articles
                USING ivfflat (summary_embedding vector_l2_ops)
                WITH (lists = 100);
            """)

            logging.info("Creating IVFFlat index on vector column for faster similarity search on topics...")
            cur.execute("""
                CREATE INDEX IF NOT EXISTS topics_embedding_idx
                ON topics
                USING ivfflat (topic_embedding vector_l2_ops)
                WITH (lists = 10);
            """)


            conn.commit()
        logging.info("Database initialized successfully.")
    except Exception as e:
        logging.error(f"Database initialization failed: {e}")
    finally:
        if conn:
            conn.close()

# ### FIX 2: The entire topic creation and assignment logic is now in this function ###
def find_or_create_topic(article_summary_embedding, article_title, article_summary):
    """
    Finds a similar topic or creates a new one, implementing the "Topic Promotion System".
    """
    if article_summary_embedding is None:
        return None, "Unthemed"

    conn = get_db_connection()
    if not conn: return None, "Unthemed"

    try:
        register_vector(conn)
        with conn.cursor() as cur:
            # 1. Find the nearest topic based on semantic similarity of summaries.
            cur.execute(
                """
                SELECT id, topic_name, topic_embedding
                FROM topics
                ORDER BY topic_embedding <-> %s
                LIMIT 1;
                """,
                (article_summary_embedding,)
            )
            nearest_topic = cur.fetchone()

            # 2. If a similar topic is found, associate with it and update its count.
            if nearest_topic:
                similarity = cosine_similarity(article_summary_embedding, nearest_topic['topic_embedding'])
                logging.info(f"Nearest topic found: '{nearest_topic['topic_name']}' with similarity {similarity:.4f}")

                if similarity >= COSINE_SIMILARITY_THRESHOLD:
                    topic_id_to_update = nearest_topic['id']
                    
                    # Increment the article count for this topic
                    cur.execute(
                        """
                        UPDATE topics
                        SET article_count = article_count + 1
                        WHERE id = %s
                        RETURNING article_count;
                        """,
                        (topic_id_to_update,)
                    )
                    new_count = cur.fetchone()['article_count']
                    
                    # Promote the topic to "visible" if it reaches the 2-article threshold.
                    if new_count >= 2:
                        cur.execute("UPDATE topics SET is_confirmed = TRUE WHERE id = %s;", (topic_id_to_update,))
                    
                    conn.commit()
                    logging.info(f"Assigned to existing topic '{nearest_topic['topic_name']}'. New count: {new_count}.")
                    return nearest_topic['id'], nearest_topic['topic_name']

            # 3. If no similar topic is found, create a new "potential" topic.
            # It will start with article_count=1 and is_confirmed=False by default.
            logging.info("No similar topic found. Creating a new potential topic.")
            new_topic_name = create_topic_name_with_gemini(article_title, article_summary)
            new_topic_embedding = article_summary_embedding

            cur.execute(
                """
                INSERT INTO topics (topic_name, topic_embedding)
                VALUES (%s, %s)
                ON CONFLICT (topic_name) DO NOTHING
                RETURNING id;
                """,
                (new_topic_name, new_topic_embedding)
            )
            
            new_topic = cur.fetchone()
            if new_topic:
                new_topic_id = new_topic['id']
            else:
                cur.execute("SELECT id FROM topics WHERE topic_name = %s", (new_topic_name,))
                new_topic_id = cur.fetchone()['id']

            conn.commit()
            logging.info(f"Created new potential topic: '{new_topic_name}' (ID: {new_topic_id})")
            return new_topic_id, new_topic_name

    except Exception as e:
        logging.error(f"Error in find_or_create_topic: {e}", exc_info=True)
        return None, "Unthemed"
    finally:
        if conn:
            conn.close()

# ==============================================================================
# --- SECTION 4: SCRAPER (from scraper.py) ---
# ==============================================================================

async def crawl_with_retry(url: str, *, model="phi3") -> tuple[str | None, str | None]:
    """Attempt to fetch content using Crawl4AI with retries, returning markdown and raw HTML."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logging.info(f"[Attempt {attempt}/{MAX_RETRIES}] Crawling: {url}")
            async with AsyncWebCrawler(llm_provider=f"ollama/{model}") as crawler:
                result = await crawler.arun(url=url, prompt=(
                    "You are an expert web scraper. Extract main content and links."
                ), raw_html=True)
            return result.markdown or "", result.html or ""
        except Exception as e:
            logging.warning(f"Attempt {attempt} failed for {url}: {e}")
            if attempt < MAX_RETRIES:
                backoff = INITIAL_BACKOFF * (2 ** (attempt - 1))
                logging.info(f"Retrying in {backoff}s...")
                await asyncio.sleep(backoff)
            else:
                logging.error(f"All retries failed for {url}")
                return None, None

def is_url_excluded(url: str) -> bool:
    """Checks if a URL should be excluded based on patterns in config."""
    for pattern in EXCLUDE_URL_PATTERNS:
        regex = re.compile(pattern.replace("*", ".*"), re.IGNORECASE)
        if regex.search(url):
            return True
    return False

def extract_links_from_markdown(markdown_text: str, base_url: str) -> list:
    """Extracts article links from markdown text generated by Crawl4AI."""
    link_pattern = re.compile(r'\[([^\]]+)\]\((https?://[^\)]+)\)')
    seen_links = set()
    articles = []
    base_domain = urlparse(base_url).netloc

    for match in link_pattern.finditer(markdown_text):
        title, link = match.groups()
        title = title.strip()
        link_domain = urlparse(link).netloc

        if title.startswith('![') or 'Learn more' in title or 'View analytics' in title:
            continue

        if base_domain in link_domain and len(title) > 20 and link not in seen_links and not is_url_excluded(link):
            seen_links.add(link)
            mock_entry = SimpleNamespace(link=link, title=title, published_parsed=None)
            articles.append((mock_entry, base_domain.replace("www.", "")))

    return articles

def scrape_homepage_for_links(base_url: str) -> list:
    """Scrapes a homepage to find links to news articles using retry logic."""
    markdown_content, _ = asyncio.run(crawl_with_retry(base_url, model="phi3")) # We only need markdown for links here
    if not markdown_content:
        logging.warning(f"Skipping homepage due to repeated failure: {base_url}")
        return []
    articles = extract_links_from_markdown(markdown_content, base_url)
    logging.info(f"Found {len(articles)} potential articles via Crawl4AI from {base_url}")
    return articles

def extract_concise_title_from_html(html_content: str, fallback_title: str) -> str:
    """
    Attempts to extract a concise, main title from an article's HTML.
    Prioritizes Open Graph, then Twitter Card, then H1/H2, then aggressively cleaned <title>.
    """
    if not html_content:
        return fallback_title

    soup = BeautifulSoup(html_content, 'lxml')

    # 1. Try Open Graph title (most reliable for concise headlines for sharing)
    og_title_tag = soup.find("meta", property="og:title")
    if og_title_tag and og_title_tag.get("content"):
        title = og_title_tag.get("content").strip()
        if 20 <= len(title) <= 150:
            logging.debug(f"Title from OG: {title}")
            return title

    # 2. Try Twitter Card title
    twitter_title_tag = soup.find("meta", attrs={"name": "twitter:title"})
    if twitter_title_tag and twitter_title_tag.get("content"):
        title = twitter_title_tag.get("content").strip()
        if 20 <= len(title) <= 150:
            logging.debug(f"Title from Twitter: {title}")
            return title

    # 3. Try common headline tags (h1, h2)
    headline_tags_to_check = ['h1', 'h2']
    headline_classes_to_check = ['title', 'entry-title', 'post-title', 'headline', 'article-title', 'post-headline']

    for tag_name in headline_tags_to_check:
        for class_name in headline_classes_to_check:
            headline_tag = soup.find(tag_name, class_=class_name)
            if headline_tag:
                text = headline_tag.get_text(strip=True)
                if 20 <= len(text) <= 150:
                    logging.debug(f"Title from {tag_name} with class {class_name}: {text}")
                    return text
        headline_tag_no_class = soup.find(tag_name)
        if headline_tag_no_class:
            text = headline_tag_no_class.get_text(strip=True)
            if 20 <= len(text) <= 150:
                logging.debug(f"Title from {tag_name} (no class): {text}")
                return text


    # 4. Fallback to the regular <title> tag with much more aggressive cleaning
    title_tag = soup.find("title")
    if title_tag:
        full_title = title_tag.get_text(strip=True)
        logging.debug(f"Attempting to clean title tag: {full_title}")

        patterns_to_remove = [
            r'\s*\|\s*.*$',
            r'\s*-\s*(Politis|CNA|Sigmalive|Cyprus Mail)\s*$',
            r'\s*-\s*([A-Za-z]+\s*News|Home|Main|Article)\s*$',
            r'\s*:\s*(Ειδήσεις|Ελλάδα|Κύπρος|Διεθνή|Οικονομία|Υγεία)\s*$',
            r'\s*(Κύπρος)\s*$',
            r' – (.*)$',
            r'\s*\(\d{1,2}/\d{1,2}/\d{4}\)$',
            r'\s*–\s*.*$'
        ]

        cleaned_title = full_title
        for pattern in patterns_to_remove:
            cleaned_title = re.sub(pattern, '', cleaned_title, flags=re.IGNORECASE).strip()

        for sep in [' - ', ' | ', ' – ']:
            if sep in cleaned_title:
                parts = [p.strip() for p in cleaned_title.split(sep)]
                if parts and len(parts[0]) > 10 and len(parts[0]) <= 150:
                    logging.debug(f"Title after splitting by '{sep}': {parts[0]}")
                    return parts[0]

        cleaned_title = re.sub(r'\s+', ' ', cleaned_title).strip()
        cleaned_title = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', cleaned_title).strip()

        if 20 <= len(cleaned_title) <= 150:
            logging.debug(f"Title after aggressive cleaning: {cleaned_title}")
            return cleaned_title
        if len(cleaned_title) > 150:
            truncated_title = cleaned_title[:147] + "..."
            logging.debug(f"Title truncated: {truncated_title}")
            return truncated_title
        if len(cleaned_title) < 20 and len(full_title) > 20:
            logging.debug(f"Cleaned title too short, using original fallback: {full_title}")
            return full_title

        logging.debug(f"Using full title as last resort: {full_title}")
        return full_title

    logging.debug(f"No title tag found, using initial fallback: {fallback_title}")
    return fallback_title


def limit_articles_per_site(articles: list, max_per_site: int = 5) -> list:
    """Limits the number of articles to process per news source."""
    domain_counts = defaultdict(int)
    filtered_articles = []
    for entry, source in articles:
        if domain_counts[source] < max_per_site:
            filtered_articles.append((entry, source))
            domain_counts[source] += 1
    return filtered_articles

def clean_markdown_content(markdown_text: str) -> str:
    """
    Cleans raw markdown to remove extraneous links, headers, and metadata
    that can confuse the LLM for summarization and categorization.
    """
    if not markdown_text:
        return ""
    
    cleaned_lines = []
    for line in markdown_text.split('\n'):
        line = line.strip()
        if re.match(r'\[([^\]]+)\]\((https?://[^\)]+)\)', line) and len(line) < 150:
            continue
        if any(keyword in line.lower() for keyword in ['facebook', 'instagram', 'linkedin', 'twitter', 'youtube', 'logo', 'subscribe', 'contact', 'premium', 'advert']):
            continue
        cleaned_lines.append(line)
    
    cleaned_content = "\n".join(cleaned_lines)
    cleaned_content = re.sub(r'\n{2,}', '\n\n', cleaned_content).strip()
    return cleaned_content


def process_article_entry(entry, source_name):
    """
    Processes a single article: fetch, AI process (summarize, categorize, topic), and store in DB.
    """
    article_url = entry.link
    initial_title = entry.title
    db_conn = get_db_connection()
    if not db_conn:
        return

    try:
        register_vector(db_conn)
        with db_conn.cursor() as cur:
            cur.execute("SELECT id FROM articles WHERE url = %s", (article_url,))
            if cur.fetchone():
                logging.info(f"Article already exists, skipping: {article_url}")
                return

        cleaned_content_raw, raw_html_content = asyncio.run(crawl_with_retry(article_url, model="phi3"))
        
        cleaned_content = clean_markdown_content(cleaned_content_raw)
        
        if not cleaned_content or len(cleaned_content.split()) < 50:
            logging.warning(f"Skipping '{initial_title}' due to insufficient or failed fetch or content length.")
            return

        final_title = extract_concise_title_from_html(raw_html_content, initial_title)
        if not final_title:
            final_title = initial_title if initial_title else "Untitled Article"
            logging.warning(f"Could not extract concise title for {article_url}. Using fallback: {final_title}")
        else:
            logging.info(f"Original title: '{initial_title}' -> Final title: '{final_title}'")

        # Step 1: Generate English summary first (most reliable)
        summaries = {}
        summaries['en'] = summarize_with_gemini(cleaned_content, lang='en')

        # Critical check: If English summary fails, we cannot proceed with translations.
        if not summaries['en']:
            logging.warning("Failed to generate English summary. Cannot generate other summaries. Falling back to truncated content.")
            summaries['en'] = ' '.join(cleaned_content.split()[:60]) + "..."
            short_summary_en = ' '.join(cleaned_content.split()[:20]) + "..."
            # No AI summaries are available, so categorize and create topic with a default prompt
            category = categorize_with_gemini(final_title, NEWS_CATEGORIES)
            summary_embedding = get_embedding(final_title)
            topic_id, topic_name = find_or_create_topic(summary_embedding, final_title, final_title)
        else:
            # Step 2: Translate English summary to other languages
            short_summary_en = summarize_with_gemini(cleaned_content, lang='en', is_short=True)
            if not short_summary_en:
                short_summary_en = ' '.join(cleaned_content.split()[:20]) + "..."

            for lang in LANGUAGES.keys():
                if lang != 'en':
                    # Use the robust English summary as the source for all translations
                    summaries[lang] = translate_with_gemini(summaries['en'], target_lang=lang)
                    if not summaries[lang]:
                        logging.warning(f"Failed to translate English summary to '{lang}'. Summary for this language will be null.")
            
            # Categorize the article based on the good English summary
            category = categorize_with_gemini(summaries['en'], NEWS_CATEGORIES)
            logging.info(f"Categorized article '{final_title}' as: {category}")
            
            # Determine Topic ID and Name based on English summary
            summary_embedding = get_embedding(summaries['en'])
            topic_id, topic_name = find_or_create_topic(summary_embedding, final_title, summaries['en'])
            # The topic name is now correctly assigned by the new logic
            logging.info(f"Article '{final_title}' assigned to topic: '{topic_name}' (ID: {topic_id})")


        is_political_article = (category == "Politics")
        image_base64 = None
        
        if is_political_article:
            logging.info(f"Political article detected. Entering strict image pipeline.")
            
            logging.info(f"Image Gen [Political-1]: AI generation from title.")
            image_base64 = generate_image_with_imagen(f"An abstract image representing the news topic of {final_title} without any recognizable faces or people.")
            
            if not image_base64:
                logging.warning(f"Image Gen [Political-1] failed. Attempting [Political-2]: AI generation from summary.")
                prompt_lang = 'el' if 'el' in summaries and summaries['el'] else 'en'
                image_base64 = generate_image_with_imagen(f"An abstract, symbolic image for the news summary: {summaries[prompt_lang]}. Do not include faces, names, or real locations.")
                
            if not image_base64:
                logging.error(f"Both AI image generation attempts failed for political article. Using static placeholder.")
                image_base64 = get_political_placeholder()

        else:
            logging.info(f"Image Gen [Step 1]: AI generation for '{final_title}' using Title.")
            image_base64 = generate_image_with_imagen(final_title)

            if not image_base64:
                logging.warning(f"Image Gen [Step 1] failed. Attempting [Step 2]: AI generation using English Summary.")
                image_base64 = generate_image_with_imagen(summaries['en'])

            if not image_base64:
                logging.warning(f"Image Gen [Step 2] failed. Attempting [Step 3]: Source image processing.")
                try:
                    if raw_html_content:
                        soup_for_image = BeautifulSoup(raw_html_content, 'lxml')
                        top_image_tag = soup_for_image.find("meta", property="og:image")
                        top_image_url = top_image_tag.get('content') if top_image_tag else None

                        if top_image_url:
                            original_image_b64 = download_and_encode_image(top_image_url)
                            if original_image_b64:
                                logging.info("Attempting [Step 4]: AI restyling of the source image.")
                                restyled_image_b64 = restyle_image_with_imagen(original_image_b64)
                                if restyled_image_b64:
                                    image_base64 = restyled_image_b64
                                    logging.info("Image Gen [Step 4] succeeded.")
                                else:
                                    logging.warning("Image Gen [Step 4] failed. Using original source image as fallback.")
                                    image_base64 = original_image_b64
                            else:
                                logging.warning(f"Could not download source image for restyling: {top_image_url}")
                    else:
                        logging.warning(f"No raw HTML content available for image processing for {article_url}.")
                except Exception as e:
                    logging.error(f"Image Gen [Step 3] (Source image processing) failed: {e}")
        
        if not image_base64:
            logging.error(f"All image generation methods failed for '{final_title}'. Using a generic placeholder.")
            image_base64 = get_generic_placeholder()

        published_time = datetime.now(pytz.timezone(CYPRUS_TIMEZONE))

        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO articles (title, url, summary_en, summary_el, summary_ru, short_summary, source, image_base64, published_date, summary_embedding, category, topic_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (url) DO UPDATE SET
                    title = EXCLUDED.title,
                    summary_en = EXCLUDED.summary_en,
                    summary_el = EXCLUDED.summary_el,
                    summary_ru = EXCLUDED.summary_ru,
                    short_summary = EXCLUDED.short_summary,
                    source = EXCLUDED.source,
                    image_base64 = EXCLUDED.image_base64,
                    published_date = EXCLUDED.published_date,
                    summary_embedding = EXCLUDED.summary_embedding,
                    category = EXCLUDED.category,
                    topic_id = EXCLUDED.topic_id;
                """,
                (final_title, article_url, summaries.get('en'), summaries.get('el'), summaries.get('ru'), short_summary_en, source_name, image_base64, published_time, summary_embedding, category, topic_id)
            )
            db_conn.commit()
            logging.info(f"Successfully stored article: '{final_title}' with category '{category}' and topic '{topic_name}'")
    except Exception as e:
        logging.error(f"CRITICAL: Failed to process article {article_url}: {e}", exc_info=True)
    finally:
        if db_conn:
            db_conn.close()

def cleanup_old_articles(source_name: str):
    """Removes older articles from a given source to keep the DB size manageable."""
    conn = get_db_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM articles
                WHERE id IN (
                    SELECT id FROM articles
                    WHERE source = %s
                    ORDER BY published_date DESC
                    OFFSET 5
                );
            """, (source_name,))
            conn.commit()
            logging.info(f"Cleaned up old articles for source: {source_name}")
    except Exception as e:
        logging.error(f"Failed to cleanup old articles for {source_name}: {e}")
    finally:
        if conn:
            conn.close()

def fetch_and_store_articles():
    """The main background task function that orchestrates the entire scraping and processing pipeline."""
    logging.info("Starting fetch_and_store_articles run...")
    all_articles = []

    for site_url in HOMEPAGE_SITES_TO_SCRAPE:
        try:
            source = urlparse(site_url).netloc.replace("www.", "")
            articles = scrape_homepage_for_links(site_url)
            all_articles.extend((entry, source) for entry, source in articles)
        except Exception as e:
            logging.error(f"Homepage fetch failed for {site_url}: {e}")
            continue

    if not all_articles:
        logging.warning("No articles found this cycle.")
        return

    all_articles = limit_articles_per_site(all_articles, max_per_site=5)
    logging.info(f"Processing {len(all_articles)} articles this run.")

    for article_data in all_articles:
        try:
            process_article_entry(*article_data)
            logging.info("Waiting 5 seconds before next article to respect API rate limits...")
            time.sleep(5)
        except Exception as e:
            logging.error(f"An error occurred while processing {article_data[0].link}: {e}")


    for source in {s for _, s in all_articles}:
        cleanup_old_articles(source)
    logging.info("Finished article fetch run.")

# ==============================================================================
# --- SECTION 5: EMAIL NOTIFICATION SYSTEM ---
# ==============================================================================

def send_welcome_email(email_address):
    """Sends a welcome email to a new subscriber."""
    if not all([SENDER_EMAIL, SENDER_PASSWORD]):
        logging.warning("SENDER_EMAIL or SENDER_PASSWORD not set. Skipping welcome email.")
        return

    msg = EmailMessage()
    msg.set_content("Thank you for subscribing to the Cyprus News Digest! You will now receive morning and evening updates with the latest news.")
    msg["Subject"] = "Welcome to the Cyprus News Digest!"
    msg["From"] = SENDER_EMAIL
    msg["To"] = email_address

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)
        logging.info(f"Welcome email sent successfully to {email_address}")
    except Exception as e:
        logging.error(f"Failed to send welcome email to {email_address}: {e}")

def send_notifications():
    """Fetches latest news and sends a digest to all subscribers."""
    if not all([SENDER_EMAIL, SENDER_PASSWORD]):
        logging.warning("SENDER_EMAIL or SENDER_PASSWORD not set. Skipping email notifications.")
        return

    conn = get_db_connection()
    if not conn: return

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT email FROM subscribers")
            subscribers = cur.fetchall()

            if not subscribers:
                logging.info("No subscribers to notify.")
                return

            # Fetch the 5 most recent articles for the digest (using English short summary)
            cur.execute("""
                SELECT a.title, a.url, a.short_summary, a.source, t.topic_name
                FROM articles AS a
                LEFT JOIN topics AS t ON a.topic_id = t.id
                ORDER BY a.published_date DESC LIMIT 5
            """)
            articles = cur.fetchall()

            if not articles:
                logging.info("No new articles to send in the notification.")
                return

            email_list = [row['email'] for row in subscribers]
            logging.info(f"Sending notifications to {len(email_list)} subscribers.")

            # Create email content
            subject = f"Your Cyprus News Digest - {datetime.now(pytz.timezone(CYPRUS_TIMEZONE)).strftime('%B %d, %Y')}"
            html_content = "<html><body><h2>Latest News from Cyprus</h2>"
            for article in articles:
                topic_display = f" ({article['topic_name']})" if article['topic_name'] else ""
                html_content += f"""
                <div style="margin-bottom: 20px; border-bottom: 1px solid #eee; padding-bottom: 15px;">
                    <h3 style="margin: 0 0 5px 0;"><a href="{article['url']}">{article['title']}</a></h3>
                    <p style="margin: 0 0 5px 0; color: #555;"><em>Source: {article['source']}{topic_display}</em></p>
                    <p style="margin: 0; color: #333;">{article['short_summary']}</p>
                </div>
                """
            html_content += "</body></html>"

            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = SENDER_EMAIL
            msg["To"] = ", ".join(email_list)
            msg.set_content("Please enable HTML to view this newsletter.")
            msg.add_alternative(html_content, subtype='html')

            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as smtp:
                smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
                smtp.send_message(msg)

            logging.info("Successfully sent all notification emails.")

    except Exception as e:
        logging.error(f"Failed to send notification emails: {e}")
    finally:
        if conn:
            conn.close()


# ==============================================================================
# --- SECTION 6: FLASK WEB APPLICATION (from app.py) ---
# ==============================================================================

app = Flask(__name__)

# --- HTML Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="{{ initial_lang }}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ localized_texts[initial_lang].cyprus_news_digest }} - PGVector Edition</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Roboto:wght@400;500&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Roboto', sans-serif; background-color: #f0f4f8; }
        .font-serif { font-family: 'Playfair Display', serif; }
        .hidden { display: none; }
        .loader {
            width: 48px; height: 48px; border: 5px solid #93c5fd;
            border-bottom-color: #1d4ed8; border-radius: 50%;
            animation: rotation 1s linear infinite;
        }
        @keyframes rotation { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .btn-similar {
            background-color: #f0f4f8; color: #374151; border: 1px solid #d1d5db;
            transition: all 0.2s ease-in-out;
        }
        .btn-similar:hover { background-color: #e5e7eb; border-color: #9ca3af; }
        .topic-btn {
            background-color: #d1fae5; /* Light green */
            color: #047857; /* Dark green */
            border: 1px solid #6ee7b7;
            transition: all 0.2s ease-in-out;
        }
        .topic-btn:hover {
            background-color: #a7f3d0;
            border-color: #34d399;
        }
        .flag-icon {
            cursor: pointer;
            width: 30px; /* Adjust size as needed */
            height: 20px;
            border: 1px solid #ccc;
            border-radius: 3px;
            object-fit: cover;
            display: inline-block;
            margin: 0 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .flag-icon:hover {
            transform: scale(1.1);
        }
        .flag-icon.active {
            border: 2px solid #3b82f6; /* Highlight active flag */
            box-shadow: 0 0 8px rgba(59, 130, 246, 0.5);
        }
        .main-header-link {
            cursor: pointer;
        }
        .main-header-link:hover .title-text {
            color: #4a90e2; /* A nice blue color for hover effect */
        }
    </style>
</head>
<body class="text-gray-800">
    <div id="subscribe-overlay" class="hidden fixed inset-0 bg-gray-900 bg-opacity-75 flex items-center justify-center z-50">
        <div class="bg-white rounded-lg shadow-2xl p-8 max-w-md w-full text-center relative">
            <button id="close-subscribe-btn" class="absolute top-4 right-4 text-gray-400 hover:text-gray-600">&times;</button>
            <h2 class="font-serif text-3xl mb-4" id="subscribe-digest-text">{{ localized_texts[initial_lang].subscribe_digest }}</h2>
            <p class="text-gray-600 mb-6" id="enter-email-updates-text">{{ localized_texts[initial_lang].enter_email_updates }}</p>
            <form id="subscribe-form">
                <input type="email" id="email-input" placeholder="your.email@example.com" required class="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 mb-4">
                <button type="submit" class="w-full bg-blue-600 text-white py-2 rounded-md hover:bg-blue-700 font-semibold" id="subscribe-now-button">{{ localized_texts[initial_lang].subscribe_now }}</button>
            </form>
            <p id="subscribe-message" class="mt-4 text-sm h-5"></p>
        </div>
    </div>

    <header class="bg-gray-800 shadow-lg text-white">
        <div class="container mx-auto px-4 sm:px-6 lg:px-8 py-6 flex justify-between items-center">
            <a href="/" class="flex items-center space-x-3 main-header-link">
                <svg class="h-10 w-10 text-teal-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M12 7.5h1.5m-1.5 3h1.5m-7.5 3h7.5m-7.5 3h7.5m3-9h3.375c.621 0 1.125.504 1.125 1.125V18a2.25 2.25 0 01-2.25 2.25M16.5 7.5V18a2.25 2.25 0 002.25 2.25M16.5 7.5V4.875c0-.621-.504-1.125-1.125-1.125H4.125C3.504 3.75 3 4.254 3 4.875V18a2.25 2.25 0 002.25 2.25h13.5M6 7.5h3v3H6v-3z" /></svg>
                <div>
                    <h1 class="font-serif text-2xl md:text-3xl tracking-wider title-text" id="cyprus-news-digest-title">{{ localized_texts[initial_lang].cyprus_news_digest }}</h1>
                    <p class="text-sm text-gray-400" id="ai-summaries-search-text">{{ localized_texts[initial_lang].ai_summaries_search }}</p>
                </div>
            </a>
            <div class="flex items-center space-x-4">
                <div class="flex space-x-2">
                    <img src="https://flagcdn.com/gb.svg" alt="English" class="flag-icon" data-lang="en">
                    <img src="https://flagcdn.com/gr.svg" alt="Ελληνικά" class="flag-icon" data-lang="el">
                    <img src="https://flagcdn.com/ru.svg" alt="Русский" class="flag-icon" data-lang="ru">
                </div>
                <button id="show-subscribe-btn" class="bg-teal-500 hover:bg-teal-600 text-white font-bold py-2 px-4 rounded-lg transition-colors">{{ localized_texts[initial_lang].subscribe_button }}</button>
            </div>
        </div>
    </header>

    <main class="container mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <div class="mb-12 max-w-2xl mx-auto">
            <div class="relative">
                <div class="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-4"><svg class="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M9 3.5a5.5 5.5 0 100 11 5.5 5.5 0 000-11zM2 9a7 7 0 1112.452 4.391l3.328 3.329a.75.75 0 11-1.06 1.06l-3.329-3.328A7 7 0 012 9z" clip-rule="evenodd" /></svg></div>
                <input type="search" id="search-bar" placeholder="{{ localized_texts[initial_lang].search_placeholder }}" class="w-full py-3 pl-11 pr-4 rounded-full bg-white border-gray-300 shadow-md focus:outline-none focus:ring-2 focus:ring-blue-500">
            </div>
            <div class="mt-6 text-center">
                <h3 class="text-lg font-semibold mb-3" id="browse-by-category-text">{{ localized_texts[initial_lang].browse_by_category }}</h3>
                <div id="category-buttons" class="flex flex-wrap justify-center gap-2">
                    </div>
                <h3 class="text-lg font-semibold mt-6 mb-3" id="browse-by-topic-text">{{ localized_texts[initial_lang].browse_by_topic }}</h3>
                <div id="topic-buttons" class="flex flex-wrap justify-center gap-2">
                    </div>
            </div>
        </div>

        <div id="results-view" class="hidden">
            <div class="flex justify-between items-center mb-8">
                <h2 id="results-title" class="text-3xl font-serif text-gray-900 border-b-2 border-teal-400 pb-2"></h2>
                <button id="back-to-main" class="bg-gray-200 hover:bg-gray-300 text-gray-800 font-bold py-2 px-4 rounded-lg transition-colors">{{ localized_texts[initial_lang].back_to_all_news }}</button>
            </div>
            <div id="results-container" class="grid gap-x-8 gap-y-12 lg:grid-cols-3 md:grid-cols-2"></div>
            <div id="results-loading" class="text-center py-16 hidden"><div class="loader mx-auto"></div></div>
        </div>

        <div id="default-view">
            <section>
                <h2 class="text-3xl font-serif text-gray-900 border-b-2 border-teal-400 pb-2 mb-8" id="todays-news-title">{{ localized_texts[initial_lang].todays_news }}</h2>
                <div id="todays-news-container" class="grid gap-x-8 gap-y-12 lg:grid-cols-3 md:grid-cols-2"></div>
                <div id="todays-loading" class="text-center py-16"><div class="loader mx-auto"></div></div>
            </section>
            <section class="mt-20">
                <h2 class="text-3xl font-serif text-gray-900 border-b-2 border-teal-400 pb-2 mb-8" id="previous-articles-title">{{ localized_texts[initial_lang].previous_articles }}</h2>
                <div id="previous-news-container" class="grid gap-x-8 gap-y-12 lg:grid-cols-3 md:grid-cols-2"></div>
                <div id="previous-loading" class="text-center py-16"><div class="loader mx-auto"></div></div>
            </section>
        </div>
    </main>

    <footer class="bg-gray-800 text-center mt-24 py-8 text-gray-400">
        <p id="footer-text">{{ localized_texts[initial_lang].footer_text }}</p>
    </footer>

    <script>
        // Pass initial language and localized texts from Flask to JavaScript
        const initialLang = "{{ initial_lang }}";
        const localizedTexts = {{ localized_texts_json | safe }};
        const localizedCategories = {{ localized_categories_json | safe }};

        let currentLang = initialLang;

        const defaultView = document.getElementById('default-view');
        const resultsView = document.getElementById('results-view');
        const searchBar = document.getElementById('search-bar');
        const backButton = document.getElementById('back-to-main');
        const subscribeOverlay = document.getElementById('subscribe-overlay');
        const showSubscribeBtn = document.getElementById('show-subscribe-btn');
        const closeSubscribeBtn = document.getElementById('close-subscribe-btn');
        const subscribeForm = document.getElementById('subscribe-form');
        const emailInput = document.getElementById('email-input');
        const subscribeMessage = document.getElementById('subscribe-message');
        const categoryButtonsContainer = document.getElementById('category-buttons');
        const topicButtonsContainer = document.getElementById('topic-buttons');

        const elementsToLocalize = {
            'subscribe-digest-text': 'subscribe_digest',
            'enter-email-updates-text': 'enter_email_updates',
            'subscribe-now-button': 'subscribe_now',
            'search-bar': 'search_placeholder',
            'browse-by-category-text': 'browse_by_category',
            'browse-by-topic-text': 'browse_by_topic',
            'back-to-main': 'back_to_all_news',
            'todays-news-title': 'todays_news',
            'previous-articles-title': 'previous_articles',
            'footer-text': 'footer_text',
            'cyprus-news-digest-title': 'cyprus_news_digest',
            'ai-summaries-search-text': 'ai_summaries_search',
            'show-subscribe-btn': 'subscribe_button',
        };

        function updateTexts(lang) {
            for (const id in elementsToLocalize) {
                const element = document.getElementById(id);
                if (element) {
                    const textKey = elementsToLocalize[id];
                    if (id === 'search-bar') {
                        element.placeholder = localizedTexts[lang][textKey];
                    } else {
                        element.textContent = localizedTexts[lang][textKey];
                    }
                }
            }
            // Update category buttons
            populateCategoryButtons();
            // Update active flag icon
            document.querySelectorAll('.flag-icon').forEach(flag => {
                if (flag.dataset.lang === lang) {
                    flag.classList.add('active');
                } else {
                    flag.classList.remove('active');
                }
            });
            // Update HTML lang attribute
            document.documentElement.lang = lang;
        }

        function getCookie(name) {
            const nameEQ = name + "=";
            const ca = document.cookie.split(';');
            for(let i=0; i < ca.length; i++) {
                let c = ca[i];
                while (c.charAt(0) === ' ') c = c.substring(1, c.length);
                if (c.indexOf(nameEQ) === 0) return c.substring(nameEQ.length, c.length);
            }
            return null;
        }

        function setCookie(name, value, days) {
            let expires = "";
            if (days) {
                const date = new Date();
                date.setTime(date.getTime() + (days*24*60*60*1000));
                expires = "; expires=" + date.toUTCString();
            }
            document.cookie = name + "=" + (value || "") + expires + "; path=/";
        }

        async function getLocalizedSummary(article, lang) {
            // Check for a pre-generated summary in the requested language
            const specificSummary = article[`summary_${lang}`];
            if (specificSummary) {
                return { text: specificSummary, note: null };
            } 
            
            // If none exists, fall back to the English summary
            const englishSummary = article.summary_en;
            if (englishSummary) {
                try {
                    // Make an API call to translate the English summary
                    const response = await fetch('/api/translate-summary', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ summary_en: englishSummary, lang: lang })
                    });
                    const result = await response.json();
                    
                    // If the translation is successful, return it with a note
                    if (response.ok) {
                        return { text: result.translation, note: localizedTexts[lang].summary_not_available_note };
                    } else {
                        // If the translation fails, log the error but do not display it
                        console.error('API translation failed:', result.error);
                    }
                } catch (error) {
                    // Log network or other errors but do not display them to the user
                    console.error('Error in on-the-fly translation:', error);
                }
                
                // Final fallback: display the English summary
                return { text: englishSummary, note: '(English summary used)' };
            }

            // Fallback for when no summary is available at all
            return { text: localizedTexts[lang].summary_not_available, note: null };
        }

        function getLocalizedCategory(category, lang) {
            const index = localizedCategories['en'].indexOf(category);
            if (index !==-1 && localizedCategories[lang] && localizedCategories[lang][index]) {
                return localizedCategories[lang][index];
            }
            return category; // Fallback to original if translation not found
        }

        async function createArticleCard(article) {
            const formattedDate = new Date(article.published_date).toLocaleString(currentLang === 'el' ? 'el-GR' : currentLang === 'ru' ? 'ru-RU' : 'en-GB', {
                day: 'numeric', month: 'long', year: 'numeric', hour: '2-digit', minute: '2-digit'
            });
            const imageHtml = article.image_base64
                ? `<div class="overflow-hidden"><img src="data:image/png;base64,${article.image_base64}" alt="AI image for ${article.title}" class="w-full h-48 object-cover group-hover:scale-105 transition-transform duration-300"></div>`
                : `<div class="h-48 bg-gray-200 flex items-center justify-center"><p class="text-sm text-gray-400">${localizedTexts[currentLang].image_not_available}</p></div>`;

            const summaryResult = await getLocalizedSummary(article, currentLang);
            const mainSummary = summaryResult.text.replace(/\\n/g, '<br>');
            const summaryNote = summaryResult.note ? `<p class="text-xs font-medium text-gray-500 mt-1">${summaryResult.note}</p>` : '';
            const mainSummaryHtml = `<p class="text-gray-600 mt-2 mb-3 leading-relaxed text-sm">${mainSummary}</p>${summaryNote}`;

            const topicHtml = article.topic_name
                ? `<p class="text-xs font-semibold text-green-600 mb-1">Topic: ${article.topic_name}</p>`
                : '';

            return `
                <div class="bg-white rounded-lg shadow-lg hover:shadow-2xl transition-shadow duration-300 flex flex-col overflow-hidden group">
                    ${imageHtml}
                    <div class="p-6 flex-grow flex flex-col">
                        <p class="text-sm font-medium text-teal-600 mb-2">${article.source || localizedTexts[currentLang].article_not_available}</p>
                        <h3 class="text-lg font-bold text-gray-900">
                            <a href="${article.url}" target="_blank" rel="noopener noreferrer" class="hover:text-blue-700">${article.title}</a>
                        </h3>
                        <p class="text-xs font-semibold text-blue-600 mb-2">${getLocalizedCategory(article.category, currentLang) || localizedTexts[currentLang].uncategorized}</p>
                        ${topicHtml} ${mainSummaryHtml}
                        <div class="mt-auto pt-4 flex justify-between items-center">
                            <p class="text-xs text-gray-400">${formattedDate}</p>
                            <button class="btn-similar text-xs font-semibold py-1 px-3 rounded-md find-similar-btn" data-id="${article.id}">${localizedTexts[currentLang].find_similar}</button>
                        </div>
                    </div>
                </div>`;
        }

        async function fetchAndDisplay(endpoint, containerId, loadingId, messageKey) {
            const container = document.getElementById(containerId);
            const loading = document.getElementById(loadingId);
            if(loading) loading.classList.remove('hidden');
            if(container) container.innerHTML = '';
            try {
                // Pass current language as a query parameter for API fetching
                const response = await fetch(`${endpoint}&lang=${currentLang}`);
                if (!response.ok) throw new Error(localizedTexts[currentLang].network_error);
                const articles = await response.json();
                if (articles.length === 0) {
                    container.innerHTML = `<p class="text-center text-gray-500 col-span-full">${localizedTexts[currentLang][messageKey]}</p>`;
                } else {
                    const articlePromises = articles.map(createArticleCard);
                    const articleCards = await Promise.all(articlePromises);
                    container.innerHTML = articleCards.join('');
                }
            } catch (error) {
                console.error('Failed to fetch:', error);
                container.innerHTML = `<p class="text-center text-red-500 col-span-full">${error.message}</p>`;
            } finally {
                if(loading) loading.classList.add('hidden');
            }
        }

        async function populateTopicButtons() {
            try {
                const response = await fetch(`/api/topics?lang=${currentLang}`);
                if (!response.ok) throw new Error(localizedTexts[currentLang].network_error);
                const topics = await response.json();
                topicButtonsContainer.innerHTML = topics.map(topic => `
                    <button class="topic-btn px-4 py-2 rounded-full text-sm font-medium transition-colors" data-topic-id="${topic.id}" data-topic-name="${topic.topic_name}">
                        ${topic.topic_name}
                    </button>
                `).join('');
            } catch (error) {
                console.error('Error populating topic buttons:', error);
                topicButtonsContainer.innerHTML = `<p class="text-center text-red-500 col-span-full">${localizedTexts[currentLang].network_error}</p>`;
            }
        }

        function showResultsView(title) {
            document.getElementById('results-title').innerText = title;
            defaultView.classList.add('hidden');
            resultsView.classList.remove('hidden');
        }

        function showDefaultView() {
            resultsView.classList.add('hidden');
            defaultView.classList.remove('hidden');
            searchBar.value = '';
            fetchAndDisplay('/api/todays-news?_=', 'todays-news-container', 'todays-loading', 'no_new_articles_today');
            fetchAndDisplay('/api/previous-news?_=', 'previous-news-container', 'previous-loading', 'no_previous_articles');
            populateTopicButtons();
        }

        let searchTimeout;
        searchBar.addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            const query = e.target.value.trim();
            if (query.length < 3 && query.length !== 0) return;
            if (query.length === 0) {
                showDefaultView();
                return;
            }
            searchTimeout = setTimeout(() => {
                showResultsView(localizedTexts[currentLang].search_results.replace('{query}', query));
                fetchAndDisplay(`/api/search?q=${encodeURIComponent(query)}&lang=${currentLang}`, 'results-container', 'results-loading', 'no_articles_search');
            }, 500);
        });

        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('find-similar-btn')) {
                const articleId = e.target.dataset.id;
                showResultsView(localizedTexts[currentLang].similar_articles_title);
                fetchAndDisplay(`/api/similar?id=${articleId}&lang=${currentLang}`, 'results-container', 'results-loading', 'no_similar_articles');
            }
        });

        backButton.addEventListener('click', showDefaultView);
        showSubscribeBtn.addEventListener('click', () => subscribeOverlay.classList.remove('hidden'));
        closeSubscribeBtn.addEventListener('click', () => subscribeOverlay.classList.add('hidden'));

        subscribeForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = emailInput.value;
            subscribeMessage.textContent = localizedTexts[currentLang].subscribing;
            subscribeMessage.className = 'mt-4 text-sm text-gray-600';
            try {
                const response = await fetch('/api/subscribe', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email: email })
                });
                const result = await response.json();
                if (response.ok) {
                    subscribeMessage.textContent = result.message || localizedTexts[currentLang].subscription_success;
                    subscribeMessage.className = 'mt-4 text-sm text-green-600';
                    emailInput.value = '';
                } else {
                    throw new Error(result.error || localizedTexts[currentLang].subscription_failed);
                }
            } catch (err) {
                subscribeMessage.textContent = err.message;
                subscribeMessage.className = 'mt-4 text-sm text-red-600';
            }
        });

        function populateCategoryButtons() {
            categoryButtonsContainer.innerHTML = localizedCategories[currentLang].map((category, index) => {
                const englishCategory = localizedCategories['en'][index];
                return `
                    <button class="category-btn bg-blue-100 text-blue-800 hover:bg-blue-200 px-4 py-2 rounded-full text-sm font-medium transition-colors" data-category-en="${englishCategory}">
                        ${category}
                    </button>
                `;
            }).join('');
        }

        categoryButtonsContainer.addEventListener('click', (e) => {
            if (e.target.classList.contains('category-btn')) {
                const categoryEn = e.target.dataset.categoryEn;
                const categoryDisplay = e.target.textContent.trim();
                showResultsView(localizedTexts[currentLang].category_title.replace('{category}', categoryDisplay));
                fetchAndDisplay(`/api/category/${encodeURIComponent(categoryEn)}?lang=${currentLang}`, 'results-container', 'results-loading', 'no_articles_category');
            }
        });

        topicButtonsContainer.addEventListener('click', (e) => {
            if (e.target.classList.contains('topic-btn')) {
                const topicId = e.target.dataset.topicId;
                const topicName = e.target.textContent.trim();
                showResultsView(localizedTexts[currentLang].topic_title.replace('{topic}', topicName));
                fetchAndDisplay(`/api/topic/${encodeURIComponent(topicId)}?lang=${currentLang}`, 'results-container', 'results-loading', 'no_articles_topic');
            }
        });

        document.querySelectorAll('.flag-icon').forEach(flagIcon => {
            flagIcon.addEventListener('click', (e) => {
                const newLang = e.target.dataset.lang;
                if (newLang !== currentLang) {
                    currentLang = newLang;
                    setCookie('lang', newLang, 365);
                    updateTexts(currentLang);
                    showDefaultView();
                }
            });
        });

        document.addEventListener('DOMContentLoaded', () => {
            const storedLang = getCookie('lang');
            if (storedLang && localizedTexts[storedLang]) {
                currentLang = storedLang;
            } else {
                currentLang = initialLang;
                setCookie('lang', currentLang, 365);
            }
            updateTexts(currentLang);
            fetchAndDisplay('/api/todays-news?_=', 'todays-news-container', 'todays-loading', 'no_new_articles_today');
            fetchAndDisplay('/api/previous-news?_=', 'previous-news-container', 'previous-loading', 'no_previous_articles');
            populateTopicButtons();
        });
    </script>
</body>
</html>
"""

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page, reading language from cookies."""
    lang = request.cookies.get('lang', DEFAULT_LANG)
    if lang not in LANGUAGES:
        lang = DEFAULT_LANG # Fallback if cookie value is invalid

    # Pass localized texts and categories as JSON strings to the template
    localized_texts_json = json.dumps(LOCALIZED_TEXTS)
    localized_categories_json = json.dumps(LOCALIZED_CATEGORIES)

    response = make_response(render_template_string(
        HTML_TEMPLATE,
        initial_lang=lang,
        localized_texts=LOCALIZED_TEXTS, # Pass Python dict directly for Jinja rendering
        localized_texts_json=localized_texts_json, # For JS access
        localized_categories_json=localized_categories_json # For JS access
    ))
    response.set_cookie('lang', lang, max_age=60*60*24*365) # Set/update cookie for 1 year
    return response

def get_start_of_today_tz():
    """Gets the start of the current day in the Cyprus timezone."""
    now_local = datetime.now(pytz.timezone(CYPRUS_TIMEZONE))
    return now_local.replace(hour=0, minute=0, second=0, microsecond=0)

def fetch_articles_from_db(query_sql, params, lang):
    conn = get_db_connection()
    if not conn: return []
    try:
        with conn.cursor() as cur:
            cur.execute(query_sql, params)
            articles = cur.fetchall()
            return articles
    except Exception as e:
        logging.error(f"Database fetch error with lang {lang}: {e}")
        return []
    finally:
        if conn:
            conn.close()

@app.route('/api/todays-news')
def get_todays_news():
    """API endpoint to fetch articles published today, localized."""
    lang = request.args.get('lang', DEFAULT_LANG)
    start_of_today = get_start_of_today_tz()
    query_sql = f"""
        SELECT a.*, t.topic_name
        FROM articles AS a
        LEFT JOIN topics AS t ON a.topic_id = t.id
        WHERE a.published_date >= %s
        ORDER BY a.published_date DESC
    """
    articles = fetch_articles_from_db(query_sql, (start_of_today,), lang)
    return jsonify(articles)

@app.route('/api/previous-news')
def get_previous_news():
    """API endpoint to fetch older articles, localized."""
    lang = request.args.get('lang', DEFAULT_LANG)
    start_of_today = get_start_of_today_tz()
    query_sql = f"""
        SELECT a.*, t.topic_name
        FROM articles AS a
        LEFT JOIN topics AS t ON a.topic_id = t.id
        WHERE a.published_date < %s
        ORDER BY a.published_date DESC
        LIMIT 51
    """
    articles = fetch_articles_from_db(query_sql, (start_of_today,), lang)
    return jsonify(articles)

@app.route('/api/search')
def search_news():
    """API endpoint for traditional full-text keyword search, localized."""
    query = request.args.get('q')
    lang = request.args.get('lang', DEFAULT_LANG)
    if not query: return jsonify([])
    search_query = ' & '.join(query.strip().split())

    query_sql = f"""
        SELECT a.*, ts_rank_cd(to_tsvector('english', a.title || ' ' || a.summary_en), query) as rank, t.topic_name
        FROM articles AS a
        LEFT JOIN topics AS t ON a.topic_id = t.id,
        to_tsquery('english', %s) query
        WHERE to_tsvector('english', a.title || ' ' || a.summary_en) @@ query
        ORDER BY rank DESC
        LIMIT 50;
    """
    articles = fetch_articles_from_db(query_sql, (search_query,), lang)
    return jsonify(articles)

@app.route('/api/category/<category_name>')
def get_articles_by_category(category_name):
    """API endpoint to fetch articles by category, localized."""
    lang = request.args.get('lang', DEFAULT_LANG)
    query_sql = f"""
        SELECT a.*, t.topic_name
        FROM articles AS a
        LEFT JOIN topics AS t ON a.topic_id = t.id
        WHERE a.category ILIKE %s
        ORDER BY a.published_date DESC
        LIMIT 50
    """
    articles = fetch_articles_from_db(query_sql, (category_name,), lang)
    return jsonify(articles)

@app.route('/api/topic/<int:topic_id>')
def get_articles_by_topic(topic_id):
    """API endpoint to fetch articles by topic ID, localized."""
    lang = request.args.get('lang', DEFAULT_LANG)
    query_sql = f"""
        SELECT a.*, t.topic_name
        FROM articles AS a
        LEFT JOIN topics AS t ON a.topic_id = t.id
        WHERE a.topic_id = %s
        ORDER BY a.published_date DESC
        LIMIT 50;
    """
    articles = fetch_articles_from_db(query_sql, (topic_id,), lang)
    return jsonify(articles)


# ### FIX 3: This API route now only shows "confirmed" topics to the user ###
@app.route('/api/topics')
def get_topics():
    """API endpoint to fetch all unique, confirmed topics for display."""
    conn = get_db_connection()
    if not conn: return jsonify([])
    try:
        with conn.cursor() as cur:
            # Only select topics that have been confirmed (2+ articles).
            cur.execute("SELECT id, topic_name FROM topics WHERE is_confirmed = TRUE ORDER BY topic_name ASC;")
            topics = cur.fetchall()
            return jsonify(topics)
    except Exception as e:
        logging.error(f"Failed to fetch topics: {e}")
        return jsonify([])
    finally:
        if conn:
            conn.close()

@app.route('/api/similar')
def find_similar_articles():
    """API endpoint for finding semantically similar articles using pgvector, localized."""
    article_id = request.args.get('id')
    lang = request.args.get('lang', DEFAULT_LANG)
    if not article_id: return jsonify({"error": "Article ID is required"}), 400

    conn = get_db_connection()
    if not conn: return jsonify([])
    try:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT summary_embedding FROM articles WHERE id = %s", (article_id,))
            source_article = cur.fetchone()
            if not source_article or source_article['summary_embedding'] is None:
                return jsonify([])

            source_embedding = source_article['summary_embedding']

            query_sql = f"""
                SELECT a.id, a.title, a.url, a.summary_en, a.summary_el, a.summary_ru, a.short_summary, a.source, a.image_base64, a.published_date, a.category, t.topic_name
                FROM articles AS a
                LEFT JOIN topics AS t ON a.topic_id = t.id
                WHERE a.id != %s AND a.summary_embedding IS NOT NULL
                ORDER BY a.summary_embedding <-> %s
                LIMIT 5;
            """
            cur.execute(query_sql, (article_id, source_embedding))
            articles = cur.fetchall()
            return jsonify(articles)
    except Exception as e:
        logging.error(f"Similarity search failed: {e}")
        return jsonify([])
    finally:
        if conn:
            conn.close()

@app.route('/api/subscribe', methods=['POST'])
def subscribe():
    """API endpoint to handle new email subscriptions."""
    data = request.get_json()
    email = data.get('email')
    if not email or not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return jsonify({"error": "Invalid email address."}), 400

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed."}), 500

    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO subscribers (email) VALUES (%s)", (email,))
            conn.commit()

        threading.Thread(target=send_welcome_email, args=(email,)).start()

        return jsonify({"message": "Thank you for subscribing! A welcome email has been sent."}), 201
    except psycopg.errors.UniqueViolation:
        return jsonify({"error": "This email is already subscribed."}), 409
    except Exception as e:
        logging.error(f"Subscription error: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/translate-summary', methods=['POST'])
def translate_summary_api():
    """API endpoint to translate an English summary on-the-fly."""
    data = request.get_json()
    english_summary = data.get('summary_en')
    target_lang = data.get('lang')

    if not english_summary or not target_lang:
        return jsonify({"error": "Missing summary or language."}), 400

    translated_summary = translate_with_gemini(english_summary, target_lang=target_lang)

    if translated_summary:
        return jsonify({"translation": translated_summary}), 200
    else:
        return jsonify({"error": "Translation failed."}), 500


# ==============================================================================
# --- SECTION 7: SCHEDULER AND APP STARTUP ---
# ==============================================================================

if __name__ == '__main__':
    # Initialize the database first
    init_db()

    scheduler = BackgroundScheduler(daemon=True, timezone=pytz.timezone(CYPRUS_TIMEZONE))

    scheduler.add_job(
        fetch_and_store_articles,
        trigger=IntervalTrigger(minutes=15),
        id='news_fetch_job',
        replace_existing=True,
        misfire_grace_time=900
    )

    scheduler.add_job(
        send_notifications,
        trigger=CronTrigger(hour='8,18', minute='0'),
        id='email_notification_job',
        replace_existing=True,
        misfire_grace_time=900
    )

    scheduler.start()

    logging.info("Scheduling initial article fetch to run in 5 seconds...")
    threading.Timer(5.0, fetch_and_store_articles).start()

    logging.info(f"Starting Cyprus News Digest on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)