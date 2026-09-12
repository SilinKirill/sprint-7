"""Telegram polling interface for the shared local RAG pipeline."""
import argparse
import asyncio
import logging
import os

from rag import Settings, RAG, ServiceError, OutputError, format_reply


def split_reply(text):
    # 1700 Unicode code points fit within 4096 UTF-16 units even with emoji.
    return [text[i:i+1700] for i in range(0, len(text), 1700)]


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    from telegram import Update
    from telegram.constants import ChatAction
    from telegram.ext import Application, CommandHandler, MessageHandler, filters

    settings = Settings.from_env()
    token = os.getenv('TELEGRAM_BOT_TOKEN', '').strip()
    if not token or token == 'PUT_TOKEN_HERE':
        raise SystemExit('Set TELEGRAM_BOT_TOKEN in task-4/.env')
    try:
        allowed = {int(x.strip()) for x in os.getenv('ALLOWED_USER_IDS', '').split(',') if x.strip()}
    except ValueError:
        raise SystemExit('ALLOWED_USER_IDS must contain numeric IDs separated by commas')
    engine = RAG(settings)

    def permitted(update):
        return update.effective_user is not None and (not allowed or update.effective_user.id in allowed)

    async def start(update: Update, context):
        if update.effective_message is None:
            return
        if not permitted(update):
            await update.effective_message.reply_text('Access to this bot is restricted.')
            return
        await update.effective_message.reply_text(
            'Ask a question about the knowledge base. I will answer in English and cite sources. '
            'Each question is processed independently.\n/id — your Telegram ID.')

    async def identity(update: Update, context):
        if update.effective_message and update.effective_user:
            await update.effective_message.reply_text(f'Telegram ID: {update.effective_user.id}')

    async def ask(update: Update, context):
        message = update.effective_message
        if message is None or not message.text:
            return
        if not permitted(update):
            await message.reply_text('Access to this bot is restricted.')
            return
        if len(message.text) > 2000:
            await message.reply_text('Please keep the question within 2000 characters.')
            return
        await context.bot.send_chat_action(chat_id=message.chat_id, action=ChatAction.TYPING)
        try:
            result = await asyncio.to_thread(engine.ask, message.text)
            reply = format_reply(result)
        except ServiceError:
            reply = 'Local model service error. Check that Ollama is running.'
        except OutputError:
            reply = 'Could not produce a valid answer with source references. Please clarify the question.'
        except ValueError:
            reply = 'Could not process the question. Please make it shorter.'
        for part in split_reply(reply):
            await message.reply_text(part, parse_mode=None)

    async def error_handler(update, context):
        # Do not print request URLs (which can contain the bot token) or chat data.
        print('Telegram error:', type(context.error).__name__, flush=True)

    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    app = Application.builder().token(token).concurrent_updates(False).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('id', identity))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, ask))
    app.add_error_handler(error_handler)
    print('Telegram bot ready. Keep this process and Ollama running. Stop with Ctrl+C.', flush=True)
    # One request at a time keeps GPU usage bounded. Polling needs no public server.
    app.run_polling(allowed_updates=['message'])


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Telegram authentication exceptions may include the token; keep them local/redacted.
        raise SystemExit(f'Bot stopped: {type(exc).__name__}. Check Ollama, few_shot.json and the local token.')
