import logging

import telebot

from src.config import LLM_MODEL, LOGGER_FORMAT, TELEGRAM_BOT_TOKEN
from src.rag_agent import RAGAgent

logging.basicConfig(
    level=logging.INFO,
    format=LOGGER_FORMAT
)
logger = logging.getLogger(__name__)

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не задан")

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

bot.set_my_commands([
    telebot.types.BotCommand("start", "Запустить бота / справка"),
    telebot.types.BotCommand("help", "Показать справку"),
    telebot.types.BotCommand("clear", "Очистить историю диалога"),
    telebot.types.BotCommand("model", "Показать текущую модель")
])

rag_agent = RAGAgent()


@bot.message_handler(commands=['start', 'help'])
def start(message):
    bot.reply_to(
        message,
        "Я бот по GPSS Studio.\n"
        "/help — справка\n"
        "/clear — очистить историю\n"
        "/model — модель",
        parse_mode='Markdown'
    )


@bot.message_handler(commands=['clear'])
def clear_command(message):
    user_id = str(message.from_user.id)
    rag_agent.clear_history(user_id)
    bot.reply_to(message, "История очищена.")


@bot.message_handler(commands=['model'])
def model_command(message):
    bot.reply_to(
        message,
        f"Модель: `{LLM_MODEL}`",
        parse_mode='Markdown'
    )


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    query = message.text.strip().replace('`', '')
    if not query:
        return

    user_id = str(message.from_user.id)
    logger.info(f"Задан вопрос от {user_id}: {query}")

    bot.send_chat_action(message.chat.id, 'typing')

    try:
        response = rag_agent.ask(user_id, query)
        bot.reply_to(message, response, parse_mode='Markdown')
    except Exception as ex:
        logger.exception(f"Ошибка при обработке запроса пользователя {user_id}. {ex}")
        bot.reply_to(message, "Произошла ошибка. Попробуйте позже.")


if __name__ == "__main__":
    logger.info("Бот запущен")
    bot.polling(
        non_stop=True,
        skip_pending=True
    )