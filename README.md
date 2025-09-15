# IMAGINATION

IMAGINE automation :)

**IMPORTANT:** All bots are configured via the `_internal/.env` file.

- To force stop a bot, make the window lose focus e.g. windows key, alt-tab, alt-esc.
- If you encounter any issues or crashes, run `_internal/debug.bat` and share the console output.

## Rebirth Bot

### Prerequisites

- Have the demon summoned and unlocked.
- Have a thread in your inventory or the skill version on your hotbar.
- Have all needed rebirth resources in your inventory e.g. apples, macca, runestones.
  - The bot attempts to pick an item from left to right, with no priority given to any item or category of item at this time.
    - For r8, I recommend forgoing small runestones due to inventory space concerns, instead using 9 medium runestones and 8 large runestones total per path (if you need any, reach out @decivolt).
  - See the below image for help planning.

![Rebirth Items](https://external-content.duckduckgo.com/iu/?u=https://drive.google.com/uc?id=1uN3Pw0trk65qLLSzgNU8tCeVXlawB_OV)

- Equip any XP gear and use your x10 demon incense BEFORE starting the bot.

### Notes
- Mitama fusion is not handled by the bot yet. For now, just run the bot again after fusing and re-summoning (don't forget to edit `IMAGINATION_REBIRTH_BOT_END_COUNTS` in `_internal/.env` if you're doing r4/r8, for example).
- The bot makes use of "close all" in "key config" > "opening and closing windows" > "special windows functions" using shift-c (not bound by default). If you don't bind that, make sure no in-game windows are covering the center pixel of the client.