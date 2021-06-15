#!/usr/bin/env python3
"""QA BOT"""


def loop_QA():
    """takes in input from the user and prints A: as a response"""
    goodbye = ['exit', 'quit', 'goodbye', 'bye']
    while True:
        print('Q:', end='')
        question = input()
        if question.lower() in goodbye:
            print('A: Goodbye')
            break
        print('A:',)


if __name__ == "__main__":
    loop_QA()
