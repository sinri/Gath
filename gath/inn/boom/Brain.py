import random
from typing import Optional

from gath.inn.boom import BoomKeywords


class Brain:

    def __init__(self):
        self.__random = random.Random()
        self.__prompt = 'masterpiece, 8k, '

    def __random_pick(self, array, is_optional: bool = True) -> Optional[str]:
        a = list(array)
        if is_optional:
            a = [None] + a
        return self.__random.choice(a)

    def boom(self):
        return {
            "model": self.random_a_model(),
            "height": 768,
            "width": 512,
            "prompt": self.random_a_prompt(),
            "negative_prompt": self.random_a_negative_prompt(),
            "steps": 25,
            "cfg": 7,
            "scheduler": "Euler a",
            "vae": "sd-vae-ft-mse-original",
            "clip_skip": 0,
            "seed": self.__random.randint(1000000, 9000000),
            "author": "boom",
        }

    def random_a_model(self):
        return self.__random_pick([
            'DarkSushiMix',
            'Hassaku',
            'MoonFilm',
            'XXMix9Realistic',
            'Kizuki',
            'BeenYouLite',
            'OnlyAnime',
            'LoliStyle-Mix-S',
        ], is_optional=False)

    def random_a_negative_prompt(self):
        return 'bad quality, worst quality, low worst, bad anatomy, deformed, bad band, bad leg, extra finger'

    def random_a_prompt(self):
        (self.__add_subject()
         .__add_subject_adj()
         .__add_hair()
         .__add_color()
         .__add_face()
         .__add_hand()
         .__add_leg()
         .__add_pose()
         )
        return self.__prompt

    def __add_subject(self):
        self.__prompt += '1girl, '
        return self

    def __add_subject_adj(self):
        t = self.__random_pick(BoomKeywords.SubjectAdj.values())
        if t is not None:
            self.__prompt += t + ", "
        return self

    def __add_hair(self):
        t = self.__random_pick(BoomKeywords.Hair.values())
        if t is not None:
            self.__prompt += t + ", "
        return self

    def __add_color(self):
        t = self.__random_pick(BoomKeywords.Color.values())
        if t is not None:
            self.__prompt += t + ", "
        return self

    def __add_head(self):
        t = self.__random_pick(BoomKeywords.Head.values())
        if t is not None:
            self.__prompt += t + ", "
        return self

    def __add_face(self):
        t = self.__random_pick(BoomKeywords.Face.values())
        if t is not None:
            self.__prompt += t + ", "
        return self

    def __add_hand(self):
        t = self.__random_pick(BoomKeywords.Hand.values())
        if t is not None:
            self.__prompt += t + ", "
        return self

    def __add_leg(self):
        t = self.__random_pick(BoomKeywords.Leg.values())
        if t is not None:
            self.__prompt += t + ", "
        return self

    def __add_pose(self):
        t = self.__random_pick(BoomKeywords.Pose.values())
        if t is not None:
            self.__prompt += t + ", "
        return self


if __name__ == '__main__':
    brain = Brain()
    print(brain.random_a_prompt())
