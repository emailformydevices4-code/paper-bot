# Setup from a phone browser (about 15 minutes, free)

## 1. Telegram alerts
1. Open Telegram, search @BotFather, send /newbot, follow the steps.
2. Copy the token it gives you (looks like 123456:ABC...).
3. Open your new bot and send it any message, e.g. "hi".
4. In your browser open: https://api.telegram.org/bot<TOKEN>/getUpdates
   Find "chat":{"id": NUMBER ...}. That NUMBER is your chat id.

## 2. GitHub repository
1. Make a free account at github.com, then New repository.
   Name it paper-bot, choose PRIVATE, tick "Add a README".
2. Add file > Create new file. Name: bot.py. Paste the contents. Commit.
3. Same for watchlist.txt.
4. Add file > Create new file. Name it exactly:
   .github/workflows/paper-bot.yml
   (typing the slashes makes the folders). Paste the contents. Commit.

## 3. Secrets
Settings > Secrets and variables > Actions > New repository secret
- TELEGRAM_TOKEN = your bot token
- TELEGRAM_CHAT_ID = your chat id number

## 4. First run
Actions tab > paper-trading-bot > Run workflow.
After ~2 minutes check your Telegram and the new state.json file in the repo.
It then runs by itself every weekday at 5 PM IST.

## Notes
- If GitHub pauses the schedule after weeks of no activity, open the Actions tab and press Enable.
- To see the full account any time: open state.json in the repo.
- Splits and bonus issues are not handled in the paper ledger.
