import sys
import time
import matplotlib.pyplot as plt
import numpy as np

# Ask y/n user input
def single_yes_or_no_question(question):
    choices = " [Y/n]: "
    reply = str(input(question + choices)).lower().strip()
    if len(reply) == 1 and reply[0] == "y":
        return True
    elif len(reply) == 1 and reply[0] == "n":
        return False
    else:
        return single_yes_or_no_question(question)

def get_string_input_stripped(presentence = ""):
    return str(input(presentence)).strip()

def print_error(error):
    _, _, exc_tb = sys.exc_info()
    print(f"Error at line {exc_tb.tb_lineno}: {error}")

def print_succes():
    print(f"\--- SUCCESSFUL\n")

def print_animation(mode = "loading"):
    if mode is "loading":
        animation = [
        "[        ]",
        "[=       ]",
        "[===     ]",
        "[====    ]",
        "[=====   ]",
        "[======  ]",
        "[====== ]",
        "[========]",
        "[ =======]",
        "[  ======]",
        "[   =====]",
        "[    ====]",
        "[     ===]",
        "[      ==]",
        "[       =]",
        "[        ]",
        "[        ]"
        ]

        notcomplete = True

        i = 0
        
        while notcomplete:
            print(animation[i % len(animation)], end='\r')
            time.sleep(.1)
            i += 1
            if i == 2*10:
                break