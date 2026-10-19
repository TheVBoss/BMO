#!/bin/bash
# Launches BMO. Replaces the old manual "activate venv, cd, screen,
# python3 main.py" steps in startup.txt with a single command.
#
# Runs main.py directly with ~/bmo_env's own python3 - no need to source
# venv/bin/activate first, and works no matter where you call it from since
# it cd's to its own location.
#
# For a persistent session that survives disconnecting SSH, wrap it in
# screen instead of running it bare:
#   screen -S bmo ~/robot_project_clean/start.sh
#   screen -r bmo      # reattach later
set -e
cd "$(dirname "$0")"
exec ~/bmo_env/bin/python3 main.py
