import pyttsx3
import multiprocessing

def speak_task(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 175)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

def speech_process_worker(text_queue):
    while True:
        text = text_queue.get()

        # Creates a new engine for each text-to-speech task.
        # At scale this is inefficient, but more than good enough
        # for this use case.
        engine = pyttsx3.init()
        engine.setProperty('rate', 175)
        engine.say(text)
        engine.runAndWait()
        del engine

class VoiceAssistant:
    def __init__(self):
        self.queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(
            target=speech_process_worker, 
            args=(self.queue,),
            daemon=True 
        )
        self.process.start()

    def speak(self, text):
        self.queue.put(text)


