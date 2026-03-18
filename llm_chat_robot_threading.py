import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32, String
from action_msgs.msg import GoalStatus

import speech_recognition as sr
import json
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play
import numpy as np
import scipy.io.wavfile as wav
import subprocess
import time
import threading
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor
from apikey import api_key

# Configuration
SILENCE_THRESHOLD = 3
FS = 16000
LISTENING_TIMEOUT = 10  # seconds
CONVERSATION_TIMEOUT = 30  # seconds of silence before going idle

client = OpenAI(api_key = api_key)

class SmartVoiceRobot(Node):
    def __init__(self):
        super().__init__("smart_voice_robot")
        
        # Publishers
        self.goal_publisher = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.eye_expression_publisher = self.create_publisher(Int32, "/eye_expression", 10)
        self.status_publisher = self.create_publisher(String, "/robot_status", 10)
        
        # Subscribers for navigation feedback
        self.create_subscription(GoalStatus, "/navigate_to_pose/_action/status", 
                               self.navigation_status_callback, 10)
        
        # State management
        self.is_listening = False
        self.is_speaking = False
        self.is_navigating = False
        self.conversation_active = False
        self.current_emotion = "neutral"
        
        # Threading and queues
        self.audio_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Locations with better structure
        self.locations = {
            "work bench": {
                "coords": (-3.95, 9.05, 0.0788, 2.3),
                "description": "The main workspace where creativity meets precision tools",
                "insight": "This is where students build, prototype, and bring their ideas to life!"
            },
            "laser": {
                "coords": (-2.3, 11.2, 0.0788, 4.00),
                "description": "Laser cutting and engraving station",
                "insight": "These machines use focused light beams to cut and engrave materials with incredible precision!"
            },
            "3d printing": {
                "coords": (7.00, 11.00, 0.0788, 4.7),
                "description": "3D printing fabrication area",
                "insight": "Layer by layer, these machines turn digital dreams into physical reality!"
            }
        }
        
        # Streamlined conversation system
        self.system_prompt = """
        You are More-Tea, a helpful robot guide in the Garage makerspace. 
        
        Response format: Always start with {emotion} tag, then your response.
        Available emotions: neutral, happy, sad, angry, confused, shocked, love, shy
        
        Available actions (call when appropriate):
        - navigate_to(location): Move to a location
        - start_tour(): Begin automated tour
        - set_emotion(emotion): Change facial expression
        
        Keep responses concise (1-2 sentences). Be friendly and helpful.
        """
        
        # Function definitions for GPT
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "navigate_to",
                    "description": "Navigate to a specific location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "enum": list(self.locations.keys())
                            }
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "start_tour",
                    "description": "Begin an automated tour of all locations",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]
        
        # Start background services
        self.start_background_services()
        
    def start_background_services(self):
        """Start all background threads"""
        # Continuous listening thread
        threading.Thread(target=self.continuous_listening_loop, daemon=True).start()
        
        # Response processing thread  
        threading.Thread(target=self.response_processing_loop, daemon=True).start()
        
        # Status monitoring thread
        threading.Thread(target=self.status_monitoring_loop, daemon=True).start()
        
        print("🤖 More-Tea is ready! Say 'Hey More-Tea' to start conversation.")
        self.set_status("idle")

    def continuous_listening_loop(self):
        """Continuously listen for wake words and commands"""
        recognizer = sr.Recognizer()
        
        with sr.Microphone(device_index=1) as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
        while rclpy.ok():
            try:
                if not self.is_speaking and not self.is_navigating:
                    with sr.Microphone(device_index=1) as source:
                        # Listen for wake word or commands
                        if not self.conversation_active:
                            audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                        else:
                            audio = recognizer.listen(source, timeout=LISTENING_TIMEOUT, phrase_time_limit=5)
                        
                        # Process audio in background
                        self.executor.submit(self.process_audio, audio)
                        
            except sr.WaitTimeoutError:
                if self.conversation_active:
                    # Timeout during conversation - check if we should end it
                    self.conversation_timeout_check()
                continue
            except Exception as e:
                print(f"Listening error: {e}")
                time.sleep(1)

    def process_audio(self, audio):
        """Process audio in background thread"""
        try:
            # Save and transcribe
            self.save_audio_data(audio, "temp_audio.wav")
            text = self.speech_to_text("temp_audio.wav")
            
            if not text:
                return
                
            print(f"👤 Heard: {text}")
            
            # Check for wake word
            if not self.conversation_active:
                if self.detect_wake_word(text):
                    self.start_conversation()
                    return
            else:
                # Add to processing queue
                self.audio_queue.put(text)
                
        except Exception as e:
            print(f"Audio processing error: {e}")

    def detect_wake_word(self, text):
        """Detect wake words to start conversation"""
        wake_words = ["hey more tea", "more tea", "hey robot", "robot"]
        text_lower = text.lower()
        return any(wake in text_lower for wake in wake_words)

    def start_conversation(self):
        """Start active conversation mode"""
        self.conversation_active = True
        self.play_notification_sound()
        self.quick_response("Hi! How can I help you?", "happy")
        
    def conversation_timeout_check(self):
        """Check if conversation should timeout"""
        # Simple timeout - can be made smarter
        time.sleep(2)
        if self.audio_queue.empty():
            self.end_conversation()
            
    def end_conversation(self):
        """End conversation mode"""
        self.conversation_active = False
        self.quick_response("I'll be here if you need me!", "neutral")
        self.set_status("idle")

    def response_processing_loop(self):
        """Process user commands and generate responses"""
        while rclpy.ok():
            try:
                if not self.audio_queue.empty():
                    user_input = self.audio_queue.get(timeout=1)
                    self.process_command_async(user_input)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Response processing error: {e}")

    def process_command_async(self, text):
        """Process command asynchronously"""
        try:
            # Quick acknowledgment
            self.set_emotion("neutral")
            
            # Generate response using GPT
            response = self.get_gpt_response(text)
            
            # Parse and execute
            self.execute_response(response)
            
        except Exception as e:
            print(f"Command processing error: {e}")
            self.quick_response("Sorry, I had trouble processing that.", "confused")

    def get_gpt_response(self, user_input):
        """Get response from GPT with function calling"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.tools,
                temperature=0.7,
                max_tokens=100
            )
            
            return response.choices[0].message
            
        except Exception as e:
            print(f"GPT API error: {e}")
            return {"content": "{neutral} Sorry, I'm having trouble thinking right now."}

    def execute_response(self, gpt_response):
        """Execute GPT response and any function calls"""
        # Handle function calls first
        if hasattr(gpt_response, 'tool_calls') and gpt_response.tool_calls:
            for tool_call in gpt_response.tool_calls:
                function_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                if function_name == "navigate_to":
                    self.navigate_to_location(args["location"])
                elif function_name == "start_tour":
                    self.start_automated_tour()
        
        # Handle text response
        if hasattr(gpt_response, 'content') and gpt_response.content:
            emotion, clean_text = self.parse_emotion_tag(gpt_response.content)
            self.quick_response(clean_text, emotion)

    def navigate_to_location(self, location):
        """Navigate to location with feedback"""
        if location not in self.locations:
            self.quick_response("I don't know where that is!", "confused")
            return
            
        self.set_status("navigating")
        self.is_navigating = True
        
        # Send navigation goal
        coords = self.locations[location]["coords"]
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x, goal.pose.position.y, goal.pose.position.z = coords[:3]
        goal.pose.orientation.z = coords[3]
        
        self.goal_publisher.publish(goal)
        self.quick_response(f"On my way to {location}!", "happy")

    def start_automated_tour(self):
        """Start automated tour with async navigation"""
        self.quick_response("Starting the grand tour!", "excited")
        
        # Run tour in background thread
        threading.Thread(target=self.execute_tour, daemon=True).start()

    def execute_tour(self):
        """Execute the tour sequence"""
        for i, (location, data) in enumerate(self.locations.items()):
            if not rclpy.ok():
                break
                
            # Navigate
            self.navigate_to_location(location)
            
            # Wait for navigation (with timeout)
            self.wait_for_navigation_complete(timeout=15)
            
            # Give insight
            self.quick_response(data["insight"], "happy")
            
            # Pause between locations
            if i < len(self.locations) - 1:
                time.sleep(3)
        
        self.quick_response("Tour complete! Hope you enjoyed it!", "love")

    def wait_for_navigation_complete(self, timeout=15):
        """Wait for navigation to complete with timeout"""
        start_time = time.time()
        while self.is_navigating and time.time() - start_time < timeout:
            time.sleep(0.5)

    def navigation_status_callback(self, msg):
        """Handle navigation status updates"""
        if msg.status == GoalStatus.STATUS_SUCCEEDED:
            self.is_navigating = False
            self.set_status("arrived")
        elif msg.status == GoalStatus.STATUS_ABORTED:
            self.is_navigating = False
            self.quick_response("I couldn't get there, sorry!", "sad")

    def quick_response(self, text, emotion="neutral"):
        """Quick text-to-speech response"""
        self.set_emotion(emotion)
        
        # Generate and play speech in background
        self.executor.submit(self.generate_and_play_speech, text)

    def generate_and_play_speech(self, text):
        """Generate and play speech"""
        self.is_speaking = True
        try:
            with client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="sage",
                input=text,
                speed=1.1
            ) as response:
                with open("speech.wav", "wb") as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)
                
                # Play audio
                audio = AudioSegment.from_wav("speech.wav")
                play(audio)
                
        except Exception as e:
            print(f"Speech generation error: {e}")
        finally:
            self.is_speaking = False

    def set_emotion(self, emotion):
        """Set robot's eye expression"""
        emotion_map = {
            "neutral": 0, "happy": 1, "sad": 2, "angry": 3,
            "confused": 4, "shocked": 5, "love": 6, "shy": 7
        }
        
        if emotion in emotion_map:
            msg = Int32()
            msg.data = emotion_map[emotion]
            self.eye_expression_publisher.publish(msg)
            self.current_emotion = emotion

    def parse_emotion_tag(self, text):
        """Parse emotion tag from response"""
        import re
        pattern = r'\{(\w+)\}'
        match = re.search(pattern, text)
        
        if match:
            emotion = match.group(1).lower()
            clean_text = re.sub(pattern, '', text).strip()
            return emotion, clean_text
        return "neutral", text

    def set_status(self, status):
        """Publish robot status"""
        msg = String()
        msg.data = status
        self.status_publisher.publish(msg)

    def status_monitoring_loop(self):
        """Monitor system status"""
        while rclpy.ok():
            # Health checks, battery monitoring, etc.
            time.sleep(5)

    # Utility methods
    def play_notification_sound(self):
        try:
            audio = AudioSegment.from_mp3("click.mp3")
            play(audio + 10)
        except:
            pass

    def save_audio_data(self, audio, filename):
        """Save audio data efficiently"""
        raw_data = audio.get_raw_data()
        audio_data = np.frombuffer(raw_data, dtype=np.int16)
        wav.write(filename, 16000, audio_data)

    def speech_to_text(self, audio_file):
        """Convert speech to text using Whisper"""
        try:
            with open(audio_file, "rb") as f:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="json"
                )
            return response.text
        except Exception as e:
            print(f"Speech-to-text error: {e}")
            return ""


def main():
    # Set audio volume
    subprocess.run(["amixer", "set", "Master", "80%"], check=False)
    
    rclpy.init()
    robot = SmartVoiceRobot()
    
    try:
        rclpy.spin(robot)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()