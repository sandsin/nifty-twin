# Nifty Twin

Finds which Asian index's session most resembles Nifty 50's since the 9:15 AM IST open, comparing each index from its own open, and updates live without a page refresh.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard (auto-updating) |
| `similarity.py` | Alignment to 9:15 IST and scoring logic |
| `market_data.py` | 1-minute data loader (Yahoo Finance via `yfinance`) |
| `test_similarity.py` | Tests on synthetic data |
| `requirements.txt` | Dependencies (Streamlit 1.50+, Plotly for the chart) |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Optional: `python test_similarity.py` to check the scoring logic (should print `OK`).

## How it works

1. Pulls 1-minute closes for Nifty (`^NSEI`) and the selected Asian indices, plus each index's previous official daily close. The last 1-minute bar can miss the closing auction (KOSPI and Nikkei by up to 0.5%), so the daily close is used.
2. Converts everything to IST and turns each index into a path of % change from its previous close. Each line starts at its opening gap. Moves are shown as % up (▲) or down (▼), both vs the previous close and since the open; only the opening gap is also shown in points. **Align by** picks how the minutes line up:
   - **Each index's own open** (default): Nifty's first N trading minutes after 9:15 IST are compared with each index's first N trading minutes after its own open. Lunch breaks are skipped, so the two lines run side by side. For example, at 11:15 IST (2h in) Hang Seng's 7:00-9:00 IST is compared with Nifty's 9:15-11:15.
   - **Same clock time**: both over the same IST minutes from 9:15.
3. Only minutes where both indices actually traded are compared. Lunch breaks and closed markets are never forward-filled.
4. Each index gets a score out of 100. The score rewards moving by about as much as Nifty, not just in the same shape:

| Component | Weight | Meaning |
|---|---|---|
| Level closeness | 35% | How close the two lines stay, opening gap included (RMSE in % points) |
| Move size match | 25% | How alike each interval's move is in size and sign. Nifty -0.27% vs -0.20% is a close match; vs +0.20% is none |
| Opening gap match | 15% | How alike the two opening gaps are |
| Shape | 15% | Do the paths have the same shape (correlation)? |
| Direction | 10% | How often the two move the same way |

   Example: Nifty opens +12 pts and falls 70, and Hang Seng opens +10 and falls 50. They score well above KOSPI opening +3% and then moving exactly like Nifty, because KOSPI's level is far away the whole time. Moves are compared in %, not raw points: 50 points on Nikkei (~40,000) is a smaller move than 50 on Nifty (~25,000).
   If a previous close is unavailable, paths start from the open and the gap component is left out.
5. Indices with less than the minimum coverage (default 70%) are listed as "partial data" and not ranked.

## Indices covered

Nikkei 225, Hang Seng, Shanghai Composite, Shenzhen Component, KOSPI, Taiwan TAIEX, Straits Times, ASX 200, Jakarta Composite, KLCI. Change the list in the sidebar or in `CANDIDATES` inside `similarity.py`.

## Sidebar settings

- **Align by**: each index's own open, or the same clock time (see above).
- **Live**: window ends now and the page auto-updates.
- **Auto-update every**: 10 to 120 seconds. Yahoo publishes 1-minute bars about once a minute, so going below 30 seconds rarely helps.
- **Trading day / Window end**: replay any of the last ~7 trading days to test how well it would have worked.
- **Move interval**: minutes per hop when comparing the size and direction of moves.
- **Min data coverage**: how much overlap is required to rank an index.

## Things to know

- **Lunch breaks (IST)**: Shanghai and Shenzhen pause 9:00-10:30, Hang Seng 9:30-10:30, Nikkei 8:00-9:00. In own-open mode they are skipped. In same-clock mode they show as gaps, or as partial data early in the window. Any gap or frozen price longer than 15 minutes counts as a break (Yahoo pads Shenzhen's lunch with flat bars).
- **Early noise**: a few minutes after the open the match is unreliable. The app warns until Nifty has about 15 bars.
- **Data lag**: free Yahoo data can lag a few minutes. The header shows Nifty's last bar time so you can see it.
- **Not trading advice**: similarity of intraday paths does not predict what Nifty does next.

## Deploy (Streamlit Community Cloud, free)

1. Create a GitHub repo and upload the files, with `app.py` at the top level.
2. Go to share.streamlit.io, sign in with GitHub, and click **Create app**.
3. Choose the repo, set the main file to `app.py`, and click **Deploy**.

Notes: the free tier sleeps after a few idle days, so open it a few minutes before 9:15. Yahoo may rate-limit shared cloud servers; if data comes back empty, run it locally.

## Ideas for next steps

- Lead/lag check (Asian markets often move before Nifty)
- Daily backtest of which index matched best on past days
- Alerts when the best match changes
