import json
import os
from enum import Enum
from typing import Optional, Union, List, Tuple

import requests


class CivitaiEntityType(Enum):
    Checkpoint = 'Checkpoint'
    TextualInversion = 'TextualInversion'
    Hypernetwork = 'Hypernetwork'
    AestheticGradient = 'AestheticGradient'
    LORA = 'LORA'
    Controlnet = 'Controlnet'
    Poses = 'Poses'


class CivitalSort(Enum):
    ByRate = 'Highest Rated'
    ByDownload = 'Most Downloaded'
    ByDate = 'Newest'


class CivitaiPeriod(Enum):
    AllTime = 'AllTime'
    Year = 'Year'
    Month = 'Month'
    Week = 'Week'
    Day = 'Day'


class CivitaiMode(Enum):
    Archived = 'Archived'
    TakenDown = 'TakenDown'


class CivitaiModelsApiResponseMetaData:
    def __init__(self, data: dict):
        self.__data = data

    def totalItems(self) -> int:
        return self.__data.get('totalItems')

    def currentPage(self) -> int:
        return self.__data.get('currentPage')

    def pageSize(self) -> int:
        return self.__data.get('pageSize')

    def totalPages(self) -> int:
        return self.__data.get('totalPages')

    def nextPage(self) -> str:
        """
        The url of the next page.
        """
        return self.__data.get('nextPage')


class CivitaiCrawler:
    """
    :see https://github.com/civitai/civitai/wiki/REST-API-Reference
    """

    def __init__(self):
        pass

    def fetch_model_list(
            self,
            limit: int = 100,
            page: int = 1,
            query: Optional[str] = None,
            tag: Optional[str] = None,
            username: Optional[str] = None,
            types: Optional[Union[List[CivitaiEntityType], Tuple[CivitaiEntityType]]] = None,
            sort: Optional[CivitalSort] = None,
            period: Optional[CivitaiPeriod] = None,
            nsfw: Optional[bool] = None
    ):
        url = 'https://civitai.com/api/v1/models'
        queries = {
            'limit': limit,
            'page': page,
        }
        if query is not None:
            queries['query'] = query
        if tag is not None:
            queries['tag'] = tag
        if username is not None:
            queries['username'] = username
        if types is not None:
            x = []
            for item in types:
                x.append(item.value)
            queries['types'] = x
        if sort is not None:
            queries['sort'] = sort.value
        if period is not None:
            queries['period'] = period.value
        if nsfw is not None:
            queries['nsfw'] = nsfw

        # print(queries)

        response = requests.get(url, params=queries)
        json = response.json()
        array = json.get('items')

        # items = []
        # for item in array:
        #     items.append(CivitaiModelEntity(item))
        #
        metadata = CivitaiModelsApiResponseMetaData(json.get('metadata'))

        return array, metadata

    def fetch_all_to_local(self, local_dir: str):
        total_page = 1
        current_page = 1

        while current_page <= total_page:
            items, metadata = self.fetch_model_list(limit=100, page=current_page)
            # total_page=metadata.totalPages()

            for item in items:
                id = item.get('id')
                if id is None:
                    print('id is None')
                    continue

                item_dir = local_dir + os.sep + f'{id}'
                if not os.path.isdir(item_dir):
                    print(f'to create dir: {item_dir}')
                    os.mkdir(item_dir)

                index_file = item_dir + os.sep + f'index.json'
                with open(index_file, mode='w', encoding='utf-8') as f:
                    json.dump(item, f)

            current_page += 1


if __name__ == '__main__':
    cc = CivitaiCrawler()
    cc.fetch_all_to_local('/Users/leqee/code/TaiyiDrawer/workspace/civitai_meta')
