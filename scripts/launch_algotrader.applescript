-- AlgoTrader launcher — double-click to start the app with no Terminal window.
-- Rebuild after editing with:
--   osacompile -o ~/Desktop/AlgoTrader.app scripts/launch_algotrader.applescript
--   /usr/libexec/PlistBuddy -c "Set :LSUIElement true" ~/Desktop/AlgoTrader.app/Contents/Info.plist
--   codesign --force --deep -s - ~/Desktop/AlgoTrader.app
--
-- LSUIElement=true (set on the compiled .app, not expressible in the script
-- itself) makes this a background agent with no Dock icon of its own — the
-- only Dock icon you'll see belongs to the actual python3/Qt window.
-- Deliberately fire-and-forget: no post-launch health check here, since an
-- unreliable timing-based check (pgrep racing the process actually forking)
-- previously caused a blocking display-alert dialog that never got dismissed,
-- leaving this applet stuck running forever. Check logs/app_launch.log by
-- hand if the app doesn't appear.

set projectDir to "/Users/jiayutee/Dev/Projects/Algorithmic_Trading_Complete_withGUI"
set pythonBin to "/Users/jiayutee/miniconda3/bin/python3"
set logFile to projectDir & "/logs/app_launch.log"

do shell script "cd " & quoted form of projectDir & " && nohup " & quoted form of pythonBin & " app.py > " & quoted form of logFile & " 2>&1 & disown"
