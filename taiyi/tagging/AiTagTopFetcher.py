import json
import random
import time
from typing import List, Dict

import requests


class AiTagTopFetcher:

    def __init__(self):
        self.__url_of_tag_v2 = 'https://api.aitag.top/tagv2'
        self.__url_of_tag_v2_get_subs = 'https://api.aitag.top/tagv2/get_subs'
        self.__shared_headers = {
            'origin': 'https://aitag.top',
            'referer': 'https://aitag.top/',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36',
        }

    def __sleep(self):
        time.sleep(random.randint(1, 3))

    def fetch_subjects(self) -> List[str]:
        response = requests.get(self.__url_of_tag_v2_get_subs, headers=self.__shared_headers)
        fetched_json = response.json()
        if fetched_json['message'] != 'success':
            raise RuntimeError()
        return fetched_json['result']

    def fetch_tags_for_subject(self, subject) -> list[Dict[str, str]]:
        """
        item is extracted from `{'desc': '热狗', 'id': 707, 'image': '', 'name': '热狗', 'sub': '食物'}`
        as `{'tag': '热狗','desc': '热狗'}`
        :param subject:
        :return: items
        """
        page = 1
        items = []
        while True:
            fetched_json = self.__fetch_tags_for_subject_page(subject, page)
            for item in fetched_json['result']:
                items.append({'tag': item['name'], 'desc': item['desc']})
            if fetched_json['page_data']['has_next']:
                page += 1
                self.__sleep()
            else:
                break
        return items

    def __fetch_tags_for_subject_page(self, subject: str, page: int):
        response = requests.post(
            self.__url_of_tag_v2,
            json={
                "method": "get_tags_from_sub",
                "sub": subject,
                "page": page
            },
            headers=self.__shared_headers
        )
        # print(response.content)
        fetched_json = response.json()
        print(fetched_json)

        # as `success`
        message = fetched_json['message']
        if message != 'success':
            raise RuntimeError()
        return fetched_json

    def download_to_json_file(self, json_file: str):
        mix = {}

        subjects = self.fetch_subjects()
        for subject in subjects:
            print(f'fetching subject {subject}...')
            items = fetcher.fetch_tags_for_subject(subject)
            mix[subject] = items
            self.__sleep()

        s = json.dumps(mix)
        with open(json_file, 'w', encoding="utf-8") as f:
            f.write(s)
            print(f"written to {json_file}")


if __name__ == '__main__':
    fetcher = AiTagTopFetcher()
    # fetcher.fetch_subjects()
    # fetcher.fetch_tags_for_subject("食物")
    fetcher.download_to_json_file('../../output/tag_dic.json')
