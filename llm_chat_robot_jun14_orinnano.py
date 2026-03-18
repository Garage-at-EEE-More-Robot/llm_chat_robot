import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32

import speech_recognition as sr
import threading
import json
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

import subprocess
import time
import random

from apikey import api_key

SILENCE_THRESHOLD = 3
FS = 16000

client = OpenAI(api_key = api_key)

# Sample trivia questions database
TRIVIA_QUESTIONS = [
    {
        "question": "What is the capital of France?",
        "answer": "paris",
        "alternatives": ["paris", "the city of paris", "paris, france"]
    },
    {
        "question": "Which planet is known as the Red Planet?",
        "answer": "mars",
        "alternatives": ["mars"]
    },
    {
        "question": "What is the largest mammal in the world?",
        "answer": "blue whale",
        "alternatives": ["blue whale", "the blue whale"]
    },
    {
        "question": "How many sides does a hexagon have?",
        "answer": "6",
        "alternatives": ["6", "six"]
    },
    {
        "question": "Who painted the Mona Lisa?",
        "answer": "leonardo da vinci",
        "alternatives": ["leonardo da vinci", "da vinci", "leonardo"]
    }
]


class TriviaChatbot(Node):
    def __init__(self):
        super().__init__("trivia_chatbot")
        self.eye_expression_publisher = self.create_publisher(
            Int32, "/eye_expression", 10
        )
        
        self.current_question = None
        self.score = 0
        self.question_count = 0
        self.max_questions = 5
        
        self.messages = [
            {
                "role": "developer",
                "content": """
                You are a fun, enthusiastic trivia game host robot named Quizzy Bot. Your job is to host a trivia game and react to the player's answers.
                
                When a player answers correctly, call the pick_up_the_reward function.
                When a player answers incorrectly, call the pick_up_the_trash function.
                
                Keep the game upbeat and encouraging. After each answer, provide a brief interesting fact about the correct answer.
                
                Speak with enthusiasm: Use exclamation points and show excitement about the game.
                Be supportive: Even when players get answers wrong, be encouraging.
                
                IMPORTANT: Begin EVERY response with an emotion tag in this exact format: {feeling = emotion}, where emotion
                must be EXACTLY one of these options:
                - neutral (0: normal expression)
                - happy (1: happy expression)
                - sad (2: sad expression)
                - angry (3: angry expression)
                - confused (4: confused expression)
                - shocked (5: shocked expression)
                - love (6: love expression)
                - shy (7: shy expression)
                
                For example: "{feeling = happy} That's correct! Well done!"
                
                Only use these exact emotion words as they correspond to my available eye expressions.
                """,
            }
        ]
        
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "pick_up_the_reward",
                    "description": "Call this function when the player answers correctly.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "pick_up_the_trash",
                    "description": "Call this function when the player answers incorrectly.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_next_question",
                    "description": "Call this function to move to the next question.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
        ]
        
        self.user_id = "user"

    def pick_up_the_reward(self):
        """Function called when player gives correct answer - implement your reward logic here"""
        self.score += 1
        print(f"Reward picked up! Current score: {self.score}/{self.question_count}")
        return f"Correct! Your score is now {self.score} out of {self.question_count}."
    
    def pick_up_the_trash(self):
        """Function called when player gives incorrect answer - implement your trash logic here"""
        print(f"Trash picked up. Current score: {self.score}/{self.question_count}")
        return f"That's not correct. Your score remains {self.score} out of {self.question_count}."
    
    def ask_next_question(self):
        """Get the next trivia question"""
        if self.question_count >= self.max_questions:
            return f"Game over! Your final score is {self.score} out of {self.max_questions}. Would you like to play again?"
        
        self.current_question = random.choice(TRIVIA_QUESTIONS)
        self.question_count += 1
        
        return f"Question {self.question_count}: {self.current_question['question']}"
    
    def check_answer(self, user_answer):
        """Check if the user's answer is correct"""
        if not self.current_question:
            return None
        
        # Convert to lowercase and strip punctuation for comparison
        user_answer = user_answer.lower().strip()
        
        # Check against main answer and alternatives
        if user_answer == self.current_question['answer'] or user_answer in self.current_question['alternatives']:
            return True
        return False

    def play_audio(self, filename):
        audio = AudioSegment.from_mp3(filename)
        audio = audio + 18  # You can also use `audio = audio + x` where x is the number of dB
        play(audio)
        
    def generate_and_play_speech(self, text, voice="nova", output_format="wav"):
        # Generate the speech audio using OpenAI's API
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            speed=1.0
        )

        output_filename = f"speech.{output_format}"
        response.stream_to_file(output_filename)
        print(f"Audio saved as {output_filename}")
        
        self.play_audio(output_filename)
        
    def save_audio(self, audio, filename):
        """Save the audio to a file."""
        raw_data = audio.get_raw_data()
        audio_data = np.frombuffer(raw_data, dtype=np.int16)
        sample_rate = audio.sample_rate if hasattr(audio, 'sample_rate') else 16000
        wav.write(filename, sample_rate, audio_data)
        print(f"Audio saved to {filename}")
        
    def speech_to_text(self, audio_file):
        with open(audio_file, "rb") as audio_data:
            response = client.audio.transcriptions.create(
                model="whisper-1", file=audio_data, response_format="json"
            )
        transcript = response.text
        return transcript

    def recognize_speech(self, retries=3, save_audio=False, filename="recorded_audio.wav"):
        """Try to recognize speech with retries on failure and play a noise while listening."""
        recognizer = sr.Recognizer()

        with sr.Microphone(device_index=1) as source:
            print("🎤 Listening...")

            recognizer.adjust_for_ambient_noise(source, duration=1)

            attempt = 0
            while attempt < retries:
                try:
                    self.play_audio("click.mp3")
                    audio = recognizer.listen(source, timeout=2, phrase_time_limit=4)
                    self.save_audio(audio, filename)
                    command = self.speech_to_text(filename)
                    
                    print(f"👤 You: {command}")
                    return command

                except (sr.UnknownValueError, sr.RequestError, sr.WaitTimeoutError):
                    attempt += 1
                    print(f"Attempt {attempt} failed. Retrying...")

                    if attempt < retries:
                        self.generate_and_play_speech("Sorry, I didn't catch that. Could you say that again?")
                    elif attempt == retries:
                        self.generate_and_play_speech("I'm afraid I still couldn't understand. Maybe try speaking a bit louder or clearer?")
                        return None

    def gpt_api(self, messages, temperature=0.7):
        try:
            while True:
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=150,
                    tools=self.tools,
                )

                response = dict(completion.choices[0].message)
                if "function_call" in response:
                    del response["function_call"]
                messages.append(response)

                tool_calls = response.get("tool_calls")
                if not tool_calls:
                    break

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    print(f"{function_name} function called...")
                    
                    function_output = ""
                    if function_name == "pick_up_the_reward":
                        function_output = self.pick_up_the_reward()
                    elif function_name == "pick_up_the_trash":
                        function_output = self.pick_up_the_trash()
                    elif function_name == "ask_next_question":
                        function_output = self.ask_next_question()

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": function_output,
                        }
                    )

            return messages
        except Exception as e:
            print(e)
            return [{"role": "error", "content": str(e)}]

    def process_answer(self, text):
        # Add user's answer to messages
        self.messages.append({"role": self.user_id, "content": text})
        
        # Evaluate the answer
        is_correct = self.check_answer(text)
        
        # Let the LLM generate appropriate response based on correctness
        self.messages = self.gpt_api(self.messages)
        response_text = self.messages[-1]["content"]
        print(f"🤖 Quizzy Bot: {response_text}")
        
        # Set eye expression based on response
        self.analyze_sentiment_and_set_expression(response_text)
        
        # Speak the response
        self.generate_and_play_speech(response_text)

    def start_game(self):
        """Initialize and start the trivia game"""
        # Welcome message
        welcome_text = "Welcome to the Trivia Challenge! I'll ask you questions, and you'll earn points for correct answers. Let's begin!"
        print(f"🤖 Quizzy Bot: {welcome_text}")
        self.generate_and_play_speech(welcome_text)
        
        # Ask first question
        first_question = self.ask_next_question()
        print(f"🤖 Quizzy Bot: {first_question}")
        self.generate_and_play_speech(first_question)
        
        # Game loop
        while self.question_count <= self.max_questions and rclpy.ok():
            # Get user's answer
            answer = self.recognize_speech()
            if answer:
                # Process the answer
                self.process_answer(answer)
                
                # If we've reached the question limit, check if they want to play again
                if self.question_count >= self.max_questions:
                    # Ask if they want to play again
                    play_again = self.recognize_speech()
                    if play_again and ("yes" in play_again.lower() or "sure" in play_again.lower()):
                        # Reset game state
                        self.score = 0
                        self.question_count = 0
                        # Start new game
                        new_game_text = "Great! Let's start a new game."
                        print(f"🤖 Quizzy Bot: {new_game_text}")
                        self.generate_and_play_speech(new_game_text)
                        
                        # Ask first question of new game
                        first_question = self.ask_next_question()
                        print(f"🤖 Quizzy Bot: {first_question}")
                        self.generate_and_play_speech(first_question)
                    else:
                        # End the game
                        goodbye_text = "Thanks for playing! See you next time!"
                        print(f"🤖 Quizzy Bot: {goodbye_text}")
                        self.generate_and_play_speech(goodbye_text)
                        break
                else:
                    # Ask the next question
                    next_question = self.ask_next_question()
                    print(f"🤖 Quizzy Bot: {next_question}")
                    self.generate_and_play_speech(next_question)

    def parse_emotion_from_response(self, text):
        """Extract the emotion tag from the response text and return both emotion and clean text"""
        import re
        
        # Match the pattern {feeling = X}
        pattern = r'\{feeling\s*=\s*(\w+)\}'
        match = re.search(pattern, text)
        
        if match:
            emotion = match.group(1).lower()
            # Remove the tag from the text
            clean_text = re.sub(pattern, '', text).strip()
            return emotion, clean_text
        else:
            # Default to neutral if no tag is found
            return "neutral", text

    def analyze_sentiment_and_set_expression(self, text):
        """
        Parse the emotion tag from GPT's response and set the eye expression accordingly.
        """
        # Extract emotion tag and clean response text
        emotion, clean_text = self.parse_emotion_from_response(text)
        
        # Map emotions to eye expressions
        eye_expression_map = {
            "neutral": 0,
            "happy": 1,
            "sad": 2,
            "angry": 3,
            "confused": 4,
            "shocked": 5,
            "love": 6,
            "shy": 7,  
        }
        
        # Get expression code (default to neutral if emotion not in map)
        expression_code = eye_expression_map.get(emotion, 0)
        
        # Create and publish message
        msg = Int32()
        msg.data = expression_code
        self.eye_expression_publisher.publish(msg)
        
        print(f"Detected emotion: {emotion}")
        print(f"Eye expression set to: {msg.data}")
        
        # Return the clean text without the emotion tag
        return clean_text


def main():
    # Set headphone volume to 100% at the start
    subprocess.run(["amixer", "set", "Headphone", "100%"])

    rclpy.init()
    chatbot = TriviaChatbot()
    
    # Start the trivia game
    chatbot.start_game()
    
    rclpy.shutdown()


if __name__ == "__main__":
    main()