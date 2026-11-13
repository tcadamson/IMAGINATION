# IMAGINATION

## Getting started

1. Download IMAGINATION.zip from the [latest release](https://github.com/tcadamson/IMAGINATION/releases/latest) and extract it
2. Have the game open and logged in
3. Run the .exe

IMAGINATION updates itself as well as the bots every time it starts, or via `update`, so you only download it once.

## Running a bot

Type commands at the `>>>` prompt:

- `run` lists the bots
- `run <bot> --help` describes that bot's functions and available options
- `--check` (if available) shows what a run would do, what it costs, and what would stop it, without running it
- A bot pauses while the game is out of focus, and continues when you restore focus
- Ctrl+C in the console quits a paused bot

Each bot checks what it needs before running and will inform you if something is missing.

## Writing a command

```
run tmg --cycles 5 --collect-apples
    ^^^ ^^^^^^^^^^ ^^^^^^^^^^^^^^^^
    bot   option         flag
```

- An option takes a value after a space, e.g. `--cycles 5`
- A flag is just its name, e.g. `--collect-apples`
- Put quotes around anything with spaces in it, e.g. a typed `--plan` in mitama_socket

## Examples

| Command | What it does |
| --- | --- |
| `run rebirth --ranks 8` | Rebirths partner to rank 8 on every path |
| `run rebirth --ranks 8 --check` | Shows what this run would do and cost, and what would stop it, without running it |
| `run rebirth --ranks 4 --mitama 3 --mitama-ranks 8` | Rebirths partner to rank 4 on every path, fuses partner with a Kusi Mitama, then rebirths to rank 8 |
| `run rebirth --ranks 444448444448` | One digit per path, in order (Tiwaz to Wyrd): rank 8 on path 6 (Hagalaz) and path 12 (Wyrd), rank 4 elsewhere |
| `run rebirth --ranks 8 --end-path 12` | Rebirths partner to rank 8 and ends partner on path 12 (Wyrd) |
| `run tmg` | Infinite cycles until you stop it |
| `run tmg --cycles 10` | Only 10 cycles |
| `run tmg --collect-apples --collect-value-packs` | Additionally collects apples from the golden light, and value packs (Value Package: Kamikaze, Wedge) from the boss box |
| `run tmg --collect-boss-box` | Collects all items from the boss box, not just the default selection |
| `run tmg --false` | Goes straight to the roof, keeping the ghosts (two MAG Presser α per cycle) |
| `run demon_force` | Rolls until the items are exhausted, leaving you to accept or discard each roll |
| `run demon_force --rolls 20` | Only 20 rolls |
| `run demon_force --lbp 50` | Discards every roll that isn't Limit Break Power 50% or more |
| `run demon_force --lbp 50 --tac 40` | Keeps Limit Break Power 50% or more, TA Chance 40% or more; discards the rest |
| `run demon_force --lbp 0` | Keeps any Limit Break Power roll |
| `run demon_force --spread-resist 30` | An effect with no preexisting alias, specified by name with hyphens for spaces |
| `run mitama_socket --plan <share link>` | Allocates the plan using a [mitama planner](https://new-moon-database.streamlit.app/mitama) share link |
| `run mitama_socket --plan <share link> --check` | Shows what the plan would allocate and cost, without running it |
| `run mitama_socket --plan "A4N4 S4 A4 N4 N2S2 K2S2 N2K2 - - - - A4S4"` | Allocates the plan using my format; one mix per path, in order, `-` for none |
| `run mitama_reroll --path 3 --mix K4S4 --support 9` | Allocates 4 Kusi and 4 Saki to path 3 (Fehu), removing and allocating until Support +9 or more |
| `run mitama_reroll --path 3 --mix K4S4 --support 9 --check` | Shows the cost, the odds, and what would stop it, without running it |
| `run mitama_reroll --path 3 --support 9` | Allocates with the mix path 3 (Fehu) already holds |
| `run mitama_reroll --path 3 --mix K4S4 --support 9 --attempts 10` | Stops after 10 attempts |
| `run mitama_reroll --path 5 --strength 6 --speed 6` | Two stats; reaches both at once |
| `run mitama_reroll --path 5 --strength 6 --speed 6 --either` | Two stats; reaching either one is sufficient |

## Presets

A preset saves a command under a convenient alias you specify, so you don't have to retype it.

1. Win+R, paste `%LOCALAPPDATA%\IMAGINATION\resources\presets.json`, and press Enter
2. Add your presets as `"alias": "command"`, with a comma between them:

   ```
   {
     "r8": "run rebirth --ranks 8",
     "df_shot": "run demon_force --shot 50 --lbp 50 --tap 50"
   }
   ```

3. Save the file, then in the console, type `presets load` (or restart IMAGINATION)
4. Type the preset's alias, e.g. `r8`, to run it

Show the loaded presets with `presets list`. A preset runs exactly as saved; you can't add options after its alias. A preset whose alias collides with a preexisting command, e.g. `run`, is skipped.

## Errors
Win+R, paste `%LOCALAPPDATA%\IMAGINATION`, and press Enter. Send me your IMAGINATION.log, and an explanation of your issue, and I will look into it. Thanks!
