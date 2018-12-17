import random


class Dice:

    def __init__(self, label, faces):

        self.label = label
        self.faces = faces
        self.probability = 1/faces
        self.value = None

    def roll(self):

        self.value = random.randint(1, self.faces)

        return self.value
