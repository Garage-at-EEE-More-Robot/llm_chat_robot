# llm_chat_robot

## Overview
`llm_chat_robot` provides voice-driven robot interaction scripts powered by OpenAI models and ROS 2. The scripts listen to speech, generate responses, publish robot goals, and drive expressive behaviors.

## Repository Structure
- `llm_chat_robot.py` – baseline voice assistant + navigation/tool-calling logic
- `llm_chat_robot_threading.py` – threaded/wake-word-oriented variant
- `llm_chat_robot_jun14_orinnano.py` – trivia/game-oriented variant
- `apikey.py` – local API key loader used by scripts