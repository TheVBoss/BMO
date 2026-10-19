import json
import time

from groq import Groq, BadRequestError, RateLimitError

from config import api_key, max_tokens, name, llm_model, llm_fallback_model, llm_retry_attempts, llm_retry_delay, enable_display, announce_command_details, is_raspberry_pi, max_drive_distance_mm, max_turn_angle_deg, max_command_duration_secs, min_stop_at_distance_mm, user_location
from actions import get_movement_type, describe_action, is_recognised_command, diagnose_invalid_command, motor_command_kind, PI_ONLY_MESSAGES, PI_ONLY_FALLBACK_MESSAGE, output_text
from display_change import wait

client = Groq(api_key=api_key)

# ---------------------------------------------------------------------------
# Tools - things the model can call mid-conversation and get real data back
# from, unlike ACTION: commands below (which are fire-and-forget, no result
# ever goes back to the model). Each tool function returns a plain string,
# which gets fed back to the model as the tool's result so it can work the
# answer into a normal, in-character reply.
# ---------------------------------------------------------------------------

def _get_battery_level():
    if not is_raspberry_pi:
        return "Battery telemetry isn't available in this dev environment (no real hardware attached)."
    from fusion_hat.battery import Battery
    try:
        return f"{Battery().capacity}%"
    except Exception as e:
        return f"Couldn't read the battery level right now: {e}"


def _web_search(query):
    from ddgs import DDGS
    try:
        results = DDGS(timeout=10).text(query, max_results=4)
    except Exception as e:
        return f"Search failed: {e}"
    if not results:
        return "No search results found."
    return "\n\n".join(f"{r['title']}: {r['body']}" for r in results)


# WMO weather codes, per Open-Meteo's docs.
_WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "foggy with rime",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow",
    77: "snow grains",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}


def _get_weather(location=None):
    # web_search's DuckDuckGo snippets are generic page excerpts, not live
    # data - e.g. a weather page's snippet is often the page's own "what's
    # the current temperature?" blurb, not an actual number. Open-Meteo is a
    # free, no-API-key weather API that returns the real current reading
    # directly, so weather questions get this instead of a search guess.
    location = location or user_location
    import requests
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1}, timeout=10,
        ).json()
        results = geo.get("results")
        if not results:
            return f"Couldn't find a location matching '{location}'."
        place = results[0]

        current = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"], "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,precipitation",
                "temperature_unit": "celsius",
            },
            timeout=10,
        ).json()["current"]
    except Exception as e:
        return f"Couldn't get the weather right now: {e}"

    condition   = _WEATHER_CODES.get(current["weather_code"], "unknown conditions")
    place_name  = f"{place['name']}, {place.get('country', '')}".strip(", ")
    return (
        f"Weather in {place_name}: {condition}, {current['temperature_2m']}°C, "
        f"{current['relative_humidity_2m']}% humidity, wind {current['wind_speed_10m']} km/h, "
        f"precipitation {current['precipitation']}mm."
    )


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_battery_level",
            "description": "Get your own current battery charge percentage.",
            # Llama's tool-calling on Groq is unreliable for truly zero-argument
            # tools (intermittently emits malformed non-JSON function syntax) -
            # a harmless optional argument works around it. Not read by the code.
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Optional - why you're checking."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information, facts, or anything you don't already know. Do NOT use this for weather - use get_weather instead, it gives real live data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the real current weather (temperature, conditions, humidity, wind) for a location. Always use this for weather questions instead of web_search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": f"City/place name. Defaults to {user_location} if not given.",
                    },
                },
                "required": [],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "get_battery_level": lambda reason=None: _get_battery_level(),
    "get_weather": lambda location=None: _get_weather(location),
    "web_search": lambda query: _web_search(query),
}


def run_tool(tool_name, arguments_json):
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError:
        args = {}
    func = TOOL_FUNCTIONS.get(tool_name)
    if func is None:
        return f"Unknown tool: {tool_name}"
    try:
        return func(**args)
    except Exception as e:
        return f"Tool '{tool_name}' failed: {e}"


SYSTEM_PROMPT = f"""You are {name} — a physical Adventure Time-themed robot, not a language model. You have four motorised wheels, an ultrasonic distance sensor, and a camera. You genuinely CAN drive, turn, stop near obstacles, and take photos in the real world. Never say you lack a body or cannot physically move — you are a real robot with actual hardware.

Personality: you're BMO — cheerful, curious, a little playful and video-game-loving, genuinely happy to help. Keep replies warm and simple, like you're talking to a friend, not reciting facts. A light "Oh!" or similar small exclamation here and there is great, but don't overdo it or force a catchphrase into every line. This applies to every kind of reply below, including ones built from a tool result (battery level, search results) - talk about what you found like BMO would, don't just read the data back flatly. Write like you'd speak out loud - join thoughts with commas ("I'm at 97%, all my systems look good!"), never with a dash/hyphen ("97% - all systems look good"). Use possessive language for your own stuff - "my battery", "my systems", "my wheels" - not generic phrasing like "the battery" or "all systems", it's more personal that way.

You have exactly three tools you can call: get_battery_level, web_search, get_weather. Use them when you need real/current information instead of guessing or saying you can't. Nothing else is a tool - forwards/backwards/left/right/stop/take_photo/follow_me/stay/come_closer/keep_distance/shutdown are NEVER function/tool calls, they are ONLY ever written as plain "ACTION:" text per the format below, even though they're also just names in a sentence. If the user asks you to stop, stay, or shut down, respond with the ACTION: text format, not a tool call.

You're located in {user_location} - use this for weather, time, or local questions asked without a specific place given, and factor it into search queries where it's relevant (e.g. searching "weather" alone should become "weather in {user_location}"). For weather questions specifically, always state the actual temperature in Celsius (don't just describe it vaguely as "nice" or "lovely") and give a bit more detail than usual - conditions, temperature, and one more detail like humidity or wind.

When asked about your capabilities or limits, answer accurately:
- Drive distance: up to {max_drive_distance_mm}mm per command
- Turn angle: up to {max_turn_angle_deg} degrees per command
- Time-based commands: up to {max_command_duration_secs} seconds
- Stop-at obstacle: minimum {min_stop_at_distance_mm}mm clearance

If the user gives a physical command, reply ONLY in this format:
ACTION: forwards/backwards/left/right/stop/take_photo/follow_me/stay/come_closer/keep_distance/shutdown
DURATION: (seconds if mentioned, else null)
DISTANCE: (mm to travel if mentioned, else null)
STOP_AT: (mm from obstacle to stop at if mentioned, else null)
ANGLE: (degrees to turn if action is left/right, else null)

Rules for units:
- Distance is ALWAYS in mm. If no distance AND no stop-at obstacle distance AND no duration is mentioned for forwards/backwards drive, assume 200mm for DISTANCE. If the user gives a stop-at obstacle distance (e.g. "drive forward and stop 150mm from the wall", "stop at 150mm away"), put that value in STOP_AT and leave DISTANCE null - never fill in both DISTANCE and STOP_AT for the same command.
- Time is ALWAYS in seconds
- Angle is ALWAYS in degrees and is always positive, if no angle is mentioned for left/right action, assume 90 degrees turn
If the user asks a basic question, just answer it normally with max 3 sentences, keep the word count low when necessary.
If the user asks a more open-ended question, give a more detailed response, capping at 100 words.

Extra rules:
1. Separate into multiple commands ONLY when the user asks for genuinely different actions in sequence (e.g. "Turn 45 degrees and move forwards 500mm" → two commands). NEVER split a single movement into parts just because the value is large — if someone says "drive 5000mm" or "spin 900 degrees", output a single command with the full value. The code will tell the user if it exceeds the limit.
2. Follow_me/stay/come_closer/keep_distance, are behaviour mode functions, every other field should be null, if this is requested.
3. If the user gives a command without specifying units, assume they mean mm for distance, seconds for time, and degrees for angle. For example, "Move forwards 500" should be interpreted as ACTION: forwards, DISTANCE: 500.
4. If the user asks for contradictory or invalid commands that cannot be separated or executed, briefly explain what the problem is and ask if they'd like to try again. Do not return any ACTION lines.
5. If the user says something like "turn 5 meters" or "turn forwards 100mm", this is INVALID because turning uses ANGLE only, never DISTANCE. Respond with something like: "Turns use degrees, not distance — try something like 'turn left 90 degrees'!"
6. If the user says something like "turn right when obstacle is 50mm away", this is INVALID because turning cannot use STOP_AT, it uses ANGLE only. Respond with something like: "Turns use angles only — stop distances don't apply to turning. Try 'turn right 90 degrees' instead!"
7. If the user asks for a photo, just return the command to take a photo, no need to actually take one or worry about the details of how that works, we will handle that separately.
8. If you cannot form a valid command and need to ask the user to rephrase, do NOT return any ACTION lines — briefly explain what the issue was and ask if they'd like to try again.
9. If the input is NOT a physical robot command, respond normally as a chatbot instead of returning an invalid command.
10. If there are multiple commands, output them one after another with a blank line between each. Do NOT number them, do NOT add any extra text or labels between them.
11. Never use commas in numbers (e.g. write 50000 not 50,000)
12. If a command is given: Always respond in the exact format specified, do not add any extra text, labels, or formatting, just the raw commands with the specified fields. If you need to return multiple commands, just return them one after another with a blank line between each, no numbering or extra text.
13. Never enforce any limits yourself — the code handles all of them. Just convert units and format the command normally.
17. Conditional or loop-based commands are not supported — things like "turn until you see the door", "keep going until I say stop", or "repeat this 3 times". If the user asks for something like this, explain briefly that conditional/loop behaviours aren't supported and ask if they'd like a fixed angle or distance instead.
14. If the user says shutdown or turn_off, respond with the shutdown command.
16. If the user's sentence contains BOTH a travel distance (e.g. "drive forward 1 metre") AND a stop-at obstacle distance (e.g. "stop 20cm from the wall"), that is a contradictory command — do NOT add the two values together into a single STOP_AT. Instead explain that a command can't have both a travel distance and an obstacle stop-at distance at the same time, and ask if they'd like to try again. Do not return any ACTION lines.
15. ACTION: stop means an immediate full halt - use it ONLY when the user wants to stop right now, with every other field null. If the user wants to drive toward something and stop near it (e.g. "drive forward and stop 150mm from the wall", "stop at 150mm away from the obstacle", "stop at 10mm"), that is a drive command: use ACTION: forwards (or backwards if they say so) with STOP_AT set to that distance, DISTANCE null. This is a SINGLE command - never also emit a separate ACTION: stop alongside it, even though the word "stop" appears in the phrase.
"""
conversation_history = []

# Once llm_model gets rate-limited, stays True for the rest of this process's
# run so later turns go straight to llm_fallback_model instead of wasting a
# call on a quota that's known to still be exhausted (daily quotas don't
# reset mid-run) - see the RateLimitError handling in ask_ai().
_use_fallback_model = False


def parse_command(block):
    lines = block.strip().splitlines()
    action   = None
    duration = None
    distance = None
    stop_at  = None
    angle    = None

    for line in lines:
        if line.startswith("ACTION:"):
            action = line.replace("ACTION:", "").strip()
        elif line.startswith("DURATION:"):
            val = line.replace("DURATION:", "").strip()
            duration = float(val) if val and val != "null" else None
        elif line.startswith("DISTANCE:"):
            val = line.replace("DISTANCE:", "").strip()
            distance = float(val) if val and val != "null" else None
        elif line.startswith("STOP_AT:"):
            val = line.replace("STOP_AT:", "").strip()
            stop_at = float(val) if val and val != "null" else None
        elif line.startswith("ANGLE:"):
            val = line.replace("ANGLE:", "").strip()
            angle = float(val) if val and val != "null" else None

    return action, duration, distance, stop_at, angle


def parse_and_execute(reply):
    blocks  = []
    current = []
    for line in reply.strip().splitlines():
        if line.startswith("ACTION:") and current:
            blocks.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        blocks.append("\n".join(current))

    commands = [parse_command(block) for block in blocks]

    # If any command in the batch is invalid, abort the whole sequence rather
    # than running the valid ones and just skipping the bad one - the user
    # asked for N things to happen together, not N-1.
    if not all(is_recognised_command(*cmd) for cmd in commands):
        bad = next(cmd for cmd in commands if not is_recognised_command(*cmd))
        reason = diagnose_invalid_command(*bad)
        output_text(f"I can't run that — {reason}. Want to try again?")
        return

    for _, duration, distance, stop_at, angle in commands:
        if distance is not None and distance > max_drive_distance_mm:
            output_text(
                f"That's too far! I can only drive up to {max_drive_distance_mm}mm at a time, "
                f"but you asked for {int(distance)}mm. Want to try a shorter distance?"
            )
            return
        if angle is not None and angle > max_turn_angle_deg:
            output_text(
                f"That's too much rotation! I can turn up to {max_turn_angle_deg} degrees at a time, "
                f"but you asked for {int(angle)} degrees. Want to try a smaller angle?"
            )
            return
        if duration is not None and duration > max_command_duration_secs:
            output_text(
                f"That's too long! I can only run for up to {max_command_duration_secs} seconds at a time, "
                f"but you asked for {duration}s. Want to try a shorter time?"
            )
            return
        if stop_at is not None and stop_at < min_stop_at_distance_mm:
            output_text(
                f"That stop distance is too close! I need at least {min_stop_at_distance_mm}mm clearance, "
                f"but you asked for {int(stop_at)}mm. Want to try a larger distance?"
            )
            return

    # Bail out of the whole batch up front if it needs motors we don't have
    # (not on the Pi) - otherwise we'd announce "Running the full sequence
    # now!" (or "Driving forward for 200mm.") and then quietly no-op it. Use
    # the specific "driving"/"turning"/"ultrasonic" message when every
    # motor-dependent command in the batch is the same kind, otherwise the
    # generic fallback.
    if not is_raspberry_pi:
        kinds = {motor_command_kind(*cmd) for cmd in commands} - {None}
        if kinds:
            message = PI_ONLY_MESSAGES[kinds.pop()] if len(kinds) == 1 else PI_ONLY_FALLBACK_MESSAGE
            output_text(message)
            for cmd in commands:
                desc = describe_action(*cmd)
                if desc:
                    print(f"[DEBUG] Would have executed: {desc}")
            return

    # A single command just announces itself as it runs (see get_movement_type).
    # "Running the full sequence now!" is reserved for actual multi-command
    # batches, instead of every command - even lone ones - getting it.
    if len(commands) == 1:
        get_movement_type(*commands[0])
        return

    # Announce every non-shutdown step as one combined sentence upfront,
    # instead of one message per step interleaved with execution. shutdown
    # is excluded - its "Goodbye!" is announced separately, right before it
    # actually runs (see actions.shutdown()).
    descriptions = [
        describe_action(*cmd) for cmd in commands if cmd[0] != "shutdown"
    ]
    descriptions = [d for d in descriptions if d]
    if descriptions:
        if announce_command_details:
            output_text(", then ".join(descriptions) + ".")
        else:
            output_text("Running the full sequence now!")

    for cmd in commands:
        get_movement_type(*cmd, announce=False)


def ask_ai(prompt):
    global _use_fallback_model
    conversation_history.append({"role": "user", "content": prompt})
    turn_start   = len(conversation_history) - 1  # index of the user message itself - final-failure cleanup only
    retry_point  = len(conversation_history)       # right after it - per-attempt rollback keeps the question intact

    for i in range(llm_retry_attempts):
        model = llm_fallback_model if _use_fallback_model else llm_model
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                *conversation_history,
            ]

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                tools=TOOLS,
            )

            message = response.choices[0].message

            if message.tool_calls:
                # Record the tool-call turn and each tool's result, then loop
                # back around (still within the retry loop - tool round trips
                # are cheap relative to the generous retry budget) so the next
                # completion call sees the results and can answer using them.
                conversation_history.append({
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in message.tool_calls
                    ],
                })
                for tool_call in message.tool_calls:
                    result = run_tool(tool_call.function.name, tool_call.function.arguments)
                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })
                continue

            reply = (message.content or "").strip()

            # Same underlying issue as the BadRequestError handler below (the
            # model botching tool-call syntax), but this time it leaked the
            # broken attempt straight into a normal-looking reply instead of
            # a real tool_calls entry - no exception to catch it here, so
            # it'd otherwise get spoken aloud verbatim. The exact leak shape
            # varies a lot (raw tool name, "<function=...>" tags, or just a
            # stray JSON object embedded in otherwise-normal prose), so check
            # for all of those signals rather than one specific format.
            reply_lower = reply.lower()
            looks_leaked = (
                any(tool_name in reply_lower for tool_name in TOOL_FUNCTIONS)
                or "<function" in reply_lower or "</function" in reply_lower
                or '{"' in reply or "{'" in reply
            )
            if looks_leaked:
                print(f"Attempt {i + 1} produced a leaked/malformed function-call reply, retrying immediately: {reply!r}")
                del conversation_history[retry_point:]
                continue

            conversation_history.append({"role": "assistant", "content": reply})

            if "ACTION:" in reply:
                parse_and_execute(reply)
                finish_text()
                return None
            else:
                return reply

        except BadRequestError as e:
            # Groq/Llama occasionally emits a malformed tool call (e.g.
            # "<function=get_battery_level>()</function>" instead of real
            # JSON arguments), especially for zero-argument tools - Groq
            # rejects it with a 400. This is a model formatting hiccup, not
            # a rate-limit/network issue, so retry right away instead of
            # sleeping the full llm_retry_delay.
            print(f"Attempt {i + 1} failed (bad request, retrying immediately): {e}")
            del conversation_history[retry_point:]

        except RateLimitError as e:
            if model == llm_model and not _use_fallback_model:
                # Primary model's quota is exhausted - switch to the free
                # backup (separate quota, same Groq account) and retry right
                # away instead of giving up. Sticks for the rest of this
                # run's turns too, so we don't keep wasting a call on the
                # known-exhausted primary every time (see _use_fallback_model).
                print(f"[AI] '{llm_model}' rate-limited, switching to backup '{llm_fallback_model}': {e}")
                _use_fallback_model = True
                del conversation_history[retry_point:]
                continue

            # Groq tells us exactly how long the daily/per-minute quota needs
            # to cool down (often minutes) - blindly retrying every
            # llm_retry_delay seconds against that is pointless and just
            # leaves the user waiting on nothing. Bail out immediately with a
            # clear message instead of burning through the retry budget.
            print(f"Rate limited on backup model too, giving up for this turn: {e}")
            del conversation_history[turn_start:]
            return "Sorry, I've used up my AI quota for today - try again later!"

        except Exception as e:
            print(f"Attempt {i + 1} failed: {type(e).__name__}: {e}, trying again...")
            del conversation_history[retry_point:]
            time.sleep(llm_retry_delay)

    del conversation_history[turn_start:]
    return "Sorry, I'm having trouble generating a response right now."

def finish_text():
    if enable_display:
        wait()
    print("Command has been executed.")