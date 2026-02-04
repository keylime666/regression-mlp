from typing import Protocol

class Greeter(Protocol):
    def greet(self) -> None: ...

class Person:
    def greet(self) -> None:
        print("こんにちは！")

class Robot:
    def greet(self) -> None:
        print("ウィーン... ガシャン... こんにちは。")

def say_hello(speaker: Greeter) -> None:
    speaker.greet()

say_hello(Person())
say_hello(Robot())
