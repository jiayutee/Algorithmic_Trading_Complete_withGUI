-- AlgoTrader launcher — double-click to start the app with no Terminal window.
-- Rebuild after editing with:
--   osacompile -o ~/Desktop/AlgoTrader.app scripts/launch_algotrader.applescript

set projectDir to "/Users/jiayutee/Dev/Projects/Algorithmic_Trading_Complete_withGUI"
set pythonBin to "/Users/jiayutee/miniconda3/bin/python3"
set logFile to projectDir & "/logs/app_launch.log"

do shell script "cd " & quoted form of projectDir & " && nohup " & quoted form of pythonBin & " app.py > " & quoted form of logFile & " 2>&1 & disown"

delay 1

-- Quick sanity check: tell the user if it looks like it didn't start.
set isRunning to (do shell script "pgrep -f 'app.py' | wc -l") as integer
if isRunning is 0 then
	display alert "AlgoTrader failed to start" message "Check the log at " & logFile buttons {"OK"} default button "OK"
end if
