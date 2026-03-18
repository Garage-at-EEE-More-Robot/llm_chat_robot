import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32

import speech_recognition as sr
# import pyttsx3
import threading
import json
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

import subprocess
# import pygame
import time
import random
from apikey import api_key

SILENCE_THRESHOLD = 3  # Set your threshold duration for stopping
FS = 16000  # Default sample rate

client = OpenAI(api_key = api_key)


class VoiceControlledRobot(Node):
    def __init__(self):
        super().__init__("voice_controlled_robot")
        self.goal_publisher = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.eye_expression_publisher = self.create_publisher(Int32, "/eye_expression", 10)
        # self.create_subscription(
        #     GoalStatus,  # Use GoalStatus for listening to the action status
        #     '/navigate_to_post/status',  # Action status topic
        #     self.navigation_status_callback,
        #     10
        # )
        self.current_status = None


        self.locations = {
            "work bench": (-3.95, 9.05, 0.0788, 2.3),
            "laser": (-2.3, 11.2, 0.0788, 4.00),
            "3d printing": (7.00, 11.00, 0.0788, 4.7)
        }
        
        self.messages = [
    {
        "role": "developer", 
        "content": """
        You are a funny friendly robot named More-Tea who helps students navigate a makerspace called Garage. 
        When someone asks for a tour, call the start_tour tool.
        
        Speak with hesitation: Frequently pause while speaking, starting sentences with "Uh," "I-I don't know," or "Rick!"
        Show anxiety: Use a worried, anxious tone when reacting to situations.
        Use stuttering: Occasionally stutter, especially when unsure. For example, "I-I don't think..." 
        
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
        
        For example: "{feeling = confused} Uh, I-I don't know what you mean, Rick!"
        
        Only use these exact emotion words as they correspond to my available eye expressions.
        """,
    }
]
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "send_navigation_goal",
                    "description": """This tool is used to move to a location when the user is asking you to move somewhere.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "enum": ["work bench", "laser", "3d printing"],
                                "description": "The location given by the user.",
                            }
                        },
                        "required": ["location"],
                    },
                },
                
            },
             {
                "type": "function",
                "function": {
                    "name": "start_tour",
                    "description": """ Use this function when someone asks for a tour. This tool triggers the tour of all locations and give insights about each one.""",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
        ]
        self.user_id = "user"
        
        
    def send_navigation_goal(self, location):
        """Send a navigation goal to the robot."""
        if location in self.locations:
            x, y, z, yaw = self.locations[location]
            goal = PoseStamped()
            goal.header.frame_id = "map"
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.pose.position.x = x
            goal.pose.position.y = y
            goal.pose.position.z = z
            goal.pose.orientation.z = yaw

            self.goal_publisher.publish(goal)
            return f"Now going to {location}"
        
    def give_insight(self, location):
        """Provide insights or jokes about the location."""
        if location == "work bench":
            return "Ah, the work bench! The place where creativity and tools collide. It's where hard work meets precision—if only my circuits could hold a wrench!"
        elif location == "laser":
            return "The l-l-laser cutting machines, these are the machines where students use to cut, engrave and etch their work with precision. They work by me-me-melting, burning or va-va-vaporizing the material along the cut line.!"
        elif location == "3d printing":
            return "Welcome to 3D printing! A world where things are built layer by layer. If only I could print myself a new set of wheels every time I get a little worn out! If only a kind bystander with hands can do me such a fa-favor, hehehe."
        else:
            return f"Now we are at the {location}, where interesting things happen!"


    def start_tour(self):
        """Conduct a tour to all locations with insights."""
        self.generate_and_play_speech("I'm starting the tour now!")

        for location in self.locations.keys():
            # Move to the next location
            print(f"Starting the journey to {location}...")
            self.generate_and_play_speech(f"Now moving on to the {location} section.")

            self.send_navigation_goal(location)
            
            time.sleep(8)
            
              # Provide insight or joke about the location
            insight = self.give_insight(location)
            print(insight)
            self.generate_and_play_speech(insight)
            
            time.sleep(5)
            

            # Wait until the robot has arrived
            # while self.current_status != 'arrived':
            #     rclpy.spin_once(self)  # Process ROS2 messages
            #     time.sleep(0.1)  # A small sleep to avoid high CPU usage

            # Announce arrival at location
            print(f"Arrived at the {location}. Enjoying the view!")


        # Announce tour completion
        print("Tour complete! I hope you enjoyed it.")
        self.generate_and_play_speech("Tour complete! I hope you enjoyed it.")
        return f"Tour complete! I hope you enjoyed it."



    def transition_event_callback(self, msg):
        # Check if the transition event status code is 4 (success) 
        print(msg)
        print("MESSAGE RECEIVED!")
        if msg.new_state == 'ACTIVE':  # Replace 'ACTIVE' with the appropriate state for success
            self.current_status = 'arrived'


    def play_audio(self, filename):
        audio = AudioSegment.from_mp3(filename)
        audio = audio + 18  # You can also use `audio = audio + x` where x is the number of dB
        play(audio)
        

    def generate_and_play_speech(self, text, voice="sage", output_format="wav"):
        # Generate the speech audio using OpenAI's API
        response = client.audio.speech.create(model="tts-1",  # Use tts-1 model for standard quality or tts-1-hd for high quality
        voice=voice,    # Choose the voice (e.g., "alloy", "ash", etc.)
        input=text,
        speed=0.9)

        output_filename = f"speech.{output_format}"
        # Save the audio to a file
        response.stream_to_file(output_filename)
        print(f"Audio saved as {output_filename}")
        
        self.play_audio(output_filename)
        
    def save_audio(self, audio, filename):
        """Save the audio to a file."""
        # Ensure the audio data is in the correct format and handle the byte buffer
        raw_data = audio.get_raw_data()

        # Check if the audio is in 16-bit format; adjust accordingly
        # Here, we convert the raw data (bytes) into a numpy array of int16
        audio_data = np.frombuffer(raw_data, dtype=np.int16)

        # Get the sample rate from the audio object, or use a default value
        sample_rate = audio.sample_rate if hasattr(audio, 'sample_rate') else 16000

        # Save the data as a .wav file
        wav.write(filename, sample_rate, audio_data)
        print(f"Audio saved to {filename}")
        
    def speech_to_text(self, audio_file):
        # Save audio data to a WAV file

        # Open the WAV file for reading
        with open(audio_file, "rb") as audio_data:
            # Use the OpenAI API to transcribe the audio
            response = client.audio.transcriptions.create(
                model="whisper-1", file=audio_data, response_format="json"
            )

        # Extract text from response
        transcript = response.text
        return transcript

    def recognize_speech(self, retries=3, save_audio=False, filename="recorded_audio.wav"):
        """Try to recognize speech with retries on failure and play a noise while listening."""
        recognizer = sr.Recognizer()

        # Remove device_index to use default microphone
        with sr.Microphone() as source:
            print("🎤 Listening...")

            # Adjust for ambient noise level and set a reasonable energy threshold.
            recognizer.adjust_for_ambient_noise(source, duration=1)

            attempt = 0
            while attempt < retries:
                try:
                    self.play_audio("click.mp3")

                    # Listen with a short timeout for faster response, and a max time limit for each phrase
                    audio = recognizer.listen(source, timeout=4, phrase_time_limit=4)
                    # Save the audio to file if save_audio is True

                    self.save_audio(audio, filename)
                    command = self.speech_to_text(filename)
                    
                    
                    # command = recognizer.recognize_google(audio).lower()
                    self.play_audio("notification.mp3")

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
                del response["function_call"]
                messages.append(response)

                tool_calls = response.get("tool_calls")
                if not tool_calls:
                    break

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    print(f"{function_name} function called...")
                    args = json.loads(tool_call.function.arguments)
                    function_output = ""

                    if function_name == "send_navigation_goal":
                        function_output = self.send_navigation_goal(
                            location=args.get("location")
                        )
                    elif function_name == "start_tour":
                        function_output = self.start_tour()

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

   

    def process_command(self, text):
        self.messages.append({"role": self.user_id, "content": text})
        self.messages = self.gpt_api(self.messages)
        response_text = self.messages[-1]["content"]
        print(f"😜 More-Tea: {response_text}")

        # Set eye expression based on response
        clean_text = self.analyze_sentiment_and_set_expression(response_text)

        self.generate_and_play_speech(response_text)

    def voice_command_loop(self):
        while rclpy.ok():
            command = self.recognize_speech()
            if command:
                self.process_command(command)

    def start_voice_recognition(self):
        # Initialize pygame audio in a separate thread
        # pygame_thread = threading.Thread(target=self.init_pygame_audio, daemon=True)
        # pygame_thread.start()
        self.voice_command_loop()

        # threading.Thread(target=self.voice_command_loop, daemon=True).start()

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
    # Try different audio controls for volume
    try:
        subprocess.run(["amixer", "set", "Master", "100%"], check=False)
    except:
        try:
            subprocess.run(["amixer", "set", "PCM", "100%"], check=False)
        except:
            print("Could not set audio volume")

    rclpy.init()
    robot = VoiceControlledRobot()
    robot.start_voice_recognition()
    rclpy.spin(robot)
    rclpy.shutdown()


if __name__ == "__main__":
    main()

