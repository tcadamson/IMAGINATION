# IMAGINATION: Imagine Automation :D

**IMPORTANT:** All bots are configured via the `.env` file.

- To force stop a bot, make the window lose focus, e.g. windows key, alt+tab, alt+esc.
- If you encounter any issues or crashes, please send me your `debug.log` file.
- Please ensure the below bar is not obstructed by any in-game windows.
  - Bots requiring in-game movement will also expect your thread and center pixel of the client window to be unobstructed.

![Bar](https://external-content.duckduckgo.com/iu/?u=https://drive.google.com/uc?id=1V54-CUXLqMKLJBsZ58GY04wGli3VAEeF)

## Rebirth Bot

### Prerequisites

- Have the demon summoned and unlocked.
- Have a thread in your inventory, or the skill version on your hotbar.
- Have all needed rebirth resources in your inventory, e.g. apples, macca, runestones.
  - The bot attempts to pick an item from left to right, with no priority given to any item or category of item at this time.
    - For r8, I recommend forgoing small runestones and relics due to inventory space concerns, instead using 9 medium runestones and 8 large runestones total per path. (If you need any, reach out @decivolt)
  - See the below image for help planning.

![Rebirth Items](https://external-content.duckduckgo.com/iu/?u=https://drive.google.com/uc?id=1uN3Pw0trk65qLLSzgNU8tCeVXlawB_OV)

- Equip any XP gear and use your x10 demon incense **BEFORE** starting the bot.
- Configure the desired rebirth counts in `.env` via `IMAGINATION_REBIRTH_BOT_END_COUNTS` and end path (if desired) via `IMAGINATION_REBIRTH_BOT_END_PATH`

### Mitama Fusion
The bot will handle mitama fusion and post-mitama rebirths for you.
- Configure the mitama via `IMAGINATION_REBIRTH_BOT_MITAMA` and have this mitama in your COMP.
- Configure the post-mitama rebirth counts via `IMAGINATION_REBIRTH_BOT_MITAMA_END_COUNTS`

### Notes
- If you need to pause and resume the bot, that's okay, it will pick up where it left off.
- If x10 demon incense is present in your inventory, the bot will refresh it every 27.5 minutes of continuous operation.
  - If you need to pause and resume the bot, please remember to refresh incense prior to resuming.
- You may configure a cathedral location to always thread to via `IMAGINATION_REBIRTH_BOT_CATHEDRAL_LOCATION`

## Demon Force Bot

### Prerequisites
- Have the demon summoned.
- Have demon force items in your inventory.

### Notes
- The bot only recognizes sands (all varieties) and loops, prioritizing sands over loops.
- At this time there is no filtering logic, it will simply bring up the discard menu and wait for you to make a decision before continuing.