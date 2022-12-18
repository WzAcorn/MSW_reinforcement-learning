import os
import time
import random
import pyautogui
import keyboard
import matplotlib.pyplot as plt
import cv2
import numpy as np
import threading
import gym
from gym import spaces
from gym.utils import seeding




######################################################
class MapleEnv():
    def __init__(self):
        self.state = [0 for i in range(288)]
        self.reward = 0
        self.done = False
        self.numbertext = ["-","0","1","2","3","4","5","6","7","8","9"]
        self.score = ["","","",""]
        self.last_score = 0
        Update_state = threading.Thread(target = self.Get_State)
        Update_state.start()
        Update_reward = threading.Thread(target = self.Get_Reward)
        Update_reward.start()
        Update_done = threading.Thread(target = self.Get_Done)
        Update_done.start()
        
    def step(self, action):
        keyboard.press("right")
        if(action == 1):
            self.jump()
        
        time.sleep(0.25)
        return self.state, self.reward, self.done, False, False

   
    def render(self):
        pass

    def Get_State(self):
        while True:
            state = []
            pic = pyautogui.screenshot(region=(0, 0, 1920, 1080))
            img_frame = np.array(pic)
            img_frame  = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
            y_size, x_size = img_frame.shape
            for y in range(9):
                for x in range(16):
                    state.append(int(np.mean(img_frame[120*y:120*(y+1),120*x:120*(x+1)])))
                    state.append(int(np.std(img_frame[120*y:120*(y+1),120*x:120*(x+1)])))

            state = np.array(state)
            state = state.reshape(1,288)
            state = state.squeeze()
            self.state = state
            
    def Get_Reward(self):
        while True:
            try:
                self.score = ["","","",""]
                for i in range(0,11):
                    text = pyautogui.locateAllOnScreen(("{}.png".format(self.numbertext[i])), confidence = 0.84, region = (120,89,200,31))
                    for num in text:
                        x,y,_,_ = num
                        if x >= 120 and x < 140:
                            self.score[0] = self.numbertext[i]
                        if x >= 140 and x < 160:
                            self.score[1] = self.numbertext[i]
                        if x >= 160 and x < 180:
                            self.score[2] = self.numbertext[i]
                        if x >= 180 and x < 200:
                            self.score[3] = self.numbertext[i]

                score = int(self.score[0] + self.score[1] + self.score[2] + self.score[3])
                if self.last_score - 30 >= score and self.last_score + 30 <= score:
                    self.reward = self.last_score
                    score = int(self.last_score + score / 2)    
                else:
                    self.reward = score
                    self.last_score = score
            except:
                pass
        
    def Get_Done(self):
        while True:
            done = False
            temp = pyautogui.locateOnScreen(
                                    "doing.png", confidence = 0.73, region = (0,80,200,200))
            if temp == None:
                temp = pyautogui.locateOnScreen(
                                    "done.png", confidence = 0.73, region = (0,80,200,200))
                if temp != None:
                    done = True
        
            self.done = done
            
    def jump(self):
        keyboard.press("alt")
        keyboard.release("alt")
        return



    def reset(self):
        self.state = [0 for i in range(288)]
        self.reward = 0
        self.last_score = 0
        self.done = False
        return self.state, self.done
        
    def play_memory(self, array):
        for i in range(len(array)):
            self.step(array[i])
        



