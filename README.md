# IMAGINATION

**IMPORTANT:** All bots are configured via the `.env` file.

- To force stop a bot, make the window lose focus, e.g. windows key, alt + tab, alt + esc
- If you encounter any issues or crashes, please send me your `debug.log` file
- Please ensure the bar shown below is fully in view, i.e. not obstructed by any in-game windows or outside the screen

![Bar](https://external-content.duckduckgo.com/iu/?u=https://drive.google.com/uc?id=1V54-CUXLqMKLJBsZ58GY04wGli3VAEeF)

- If the mouse isn't dragging the camera properly, you will need to increase `IMAGINATION_BOT_SLEEP_AMOUNT` in `.env` (and possibly `IMAGINATION_BOT_DRAG_SLEEP_AMOUNT`) until it behaves correctly and consistently
- It is normal and expected for the bot to start slow and then increase in speed as the template regions are cached

## Rebirth Bot

### Prerequisites

- Have the demon summoned and unlocked
- Have a thread in your inventory, or the skill version on your hotbar, and ensure it is fully in view
- Have all needed rebirth resources in your inventory, e.g. apples, macca, runestones
  - The bot attempts to pick an item from left to right, with no priority given to any item or category of item at this time
  - For r8, I recommend forgoing small runestones and relics due to inventory space concerns, instead using 9 medium runestones and 8 large runestones total per path
- Configure the desired rebirth counts in `.env` via `IMAGINATION_REBIRTH_BOT_END_COUNTS` and end path (if desired) via `IMAGINATION_REBIRTH_BOT_END_PATH`

![Rebirth Items](https://external-content.duckduckgo.com/iu/?u=https://drive.google.com/uc?id=1uN3Pw0trk65qLLSzgNU8tCeVXlawB_OV)

### Mitama Fusion
If a mitama is specified, the bot will execute mitama fusion.
- Configure the mitama via `IMAGINATION_REBIRTH_BOT_MITAMA` and have this mitama in your COMP
- Configure the post-mitama rebirth counts via `IMAGINATION_REBIRTH_BOT_MITAMA_END_COUNTS`

### Notes
- Equip any XP gear and use your x10 demon incense **BEFORE** starting the bot
- If you need to stop and resume the bot, it will pick up where it left off
- If x10 demon incense is present in your inventory, the bot will refresh it every 27.5 minutes of **CONTINUOUS OPERATION** (you will need to manually refresh if stopping and resuming)
- You may configure a cathedral location to always thread to via `IMAGINATION_REBIRTH_BOT_CATHEDRAL_LOCATION`

## Demon Force Bot

### Prerequisites
- Have the demon summoned
- Have demon force items in your inventory

### Notes
- The bot recognizes all varieties of sands, scabbards, and loops, and prioritizes them in that order
- At this time, there is no filtering logic; it will simply bring up the discard menu and wait for you to make a decision before continuing
