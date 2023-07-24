import json
import os
import random
import time
from enum import Enum
from typing import Optional, Union, List, Tuple

import requests

from gath.kit.GathDB import GathDB


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
    ) -> Tuple[List[dict], CivitaiModelsApiResponseMetaData]:
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
        if response.status_code != 200:
            print(f'status code: {response.status_code}\n{response.text}')
            raise Exception('non 200')

        json = response.json()
        array = json.get('items')

        # items = []
        # for item in array:
        #     items.append(CivitaiModelEntity(item))
        #
        metadata = CivitaiModelsApiResponseMetaData(json.get('metadata'))

        return array, metadata

    def fetch_all_to_local(self, from_page: int = 1, local_dir: Optional[str] = None):
        total_page = from_page
        current_page = from_page

        while current_page <= total_page:
            print(f'{time.time()} | to fetch page {current_page} of {total_page}')
            try:
                items, metadata = self.fetch_model_list(limit=100, page=current_page)
            except Exception as e:
                print(f'failed: {e}')
                time.sleep(30)
                continue

            total_page = metadata.totalPages()

            item_index = 0
            for item in items:
                item_index += 1

                id = item.get('id')
                if id is None:
                    print('id is None')
                    continue

                if local_dir is not None:
                    item_dir = local_dir + os.sep + f'{id}'
                    if not os.path.isdir(item_dir):
                        # print(f'to create dir: {item_dir}')
                        os.mkdir(item_dir)

                    index_file = item_dir + os.sep + f'index.json'
                    with open(index_file, mode='w', encoding='utf-8') as f:
                        json.dump(item, f)

                print(
                    f'{time.time()} | ({item_index} of page {current_page}) refreshing model [{id}] {item.get("name")}')

                self.refresh_a_civitai_model_meta(item)

            current_page += 1
            x = random.randint(1, 5)
            print(f'{time.time()} | to sleep {x} seconds')
            time.sleep(x)

    def refresh_a_civitai_model_meta(self, data: dict):
        db = GathDB()
        tm = db.build_civitai_model_table()

        poi = data.get('poi')
        if poi is not None:
            poi = 'Y' if poi else 'N'

        nsfw = data.get('nsfw')
        if nsfw is not None:
            nsfw = 'Y' if nsfw else 'N'

        tm.replace_one_row({
            'model_id': data['id'],
            'model_name': data['name'],
            'description': data['description'],
            'type': data['type'],
            'poi': poi,
            'nsfw': nsfw,
            'allowNoCredit': data['allowNoCredit'],
            'allowCommercialUse': data['allowCommercialUse'],
            'allowDerivatives': data['allowDerivatives'],
            'allowDifferentLicense': data['allowDifferentLicense'],
            'creator_name': data['creator']['username'],
            'creator_img': data['creator']['image'],
            'refresh_time': tm.now(),
        })

        tm_tag = db.build_civitai_model_tag_table()
        tag_rows = []
        for tag in data.get('tags'):
            tag_rows.append({
                'model_id': data['id'],
                'tag': tag,
                'refresh_time': tm.now(),
            })
        if len(tag_rows) > 0:
            tm_tag.replace_many_rows_with_dicts(tag_rows)

        tm_version = db.build_civitai_model_version_table()
        tm_version_file = db.build_civitai_model_version_file_table()
        tm_image = db.build_civitai_image_table()

        model_versions = data.get('modelVersions')
        model_version_rows = []
        model_versions_files_rows = []
        image_rows = []
        for model_version in model_versions:
            model_version_rows.append({
                'version_id': model_version['id'],
                'model_id': data['id'],
                'name': model_version['name'],
                'create_time': db.tranform_tz_time_to_bj(model_version['createdAt']),
                'update_time': db.tranform_tz_time_to_bj(model_version['updatedAt']),
                'trained_words': json.dumps(model_version['trainedWords']),
                'base_model': model_version['baseModel'],
                'early_access_time_frame': model_version['earlyAccessTimeFrame'],
                'description': model_version['description'],
                'stats_download_count': model_version['stats']['downloadCount'],
                'stats_rating_count': model_version['stats']['ratingCount'],
                'stats_rating': model_version['stats']['rating'],
                'download_url': model_version.get('downloadUrl'),
                'refresh_time': tm_version.now(),
            })
            files = model_version.get('files')
            if files is not None and len(files) > 0:
                for file in files:
                    hash_value = None
                    hashes = file.get('hashes')
                    if hashes is not None:
                        hash_value = hashes.get('SHA256')
                    model_versions_files_rows.append({
                        'file_id': file['id'],
                        'file_name': file['name'],
                        'size_in_kb': file['sizeKB'],
                        'type': file['type'],
                        'metadata': json.dumps(file['metadata']),
                        'hash_sha256': hash_value,
                        'download_url': file['downloadUrl'],
                        'refresh_time': tm_version_file.now(),
                        'version_id':model_version['id'],
                        'model_id': data['id'],
                    })

            images = model_version.get('images')
            if images is not None and len(images) > 0:
                for image in images:
                    # image_meta=image.get('meta')
                    image_meta = json.dumps(image.get('meta'))

                    image_rows.append({
                        'image_id': image['id'],
                        'url': image['url'],
                        'nsfw': image['nsfw'],
                        'width': image['width'],
                        'height': image['height'],
                        'hash': image['hash'],
                        'meta': image_meta,
                        'model_id': data['id'],
                        'version_id': model_version['id'],
                        'refresh_time': tm_image.now(),
                    })

        if len(model_version_rows) > 0:
            tm_version.replace_many_rows_with_dicts(model_version_rows)
            if len(model_versions_files_rows) > 0:
                tm_version_file.replace_many_rows_with_dicts(model_versions_files_rows)
            if len(image_rows) > 0:
                tm_image.replace_many_rows_with_dicts(image_rows)


if __name__ == '__main__':
    cc = CivitaiCrawler()
    #local_dir = '/Users/leqee/code/TaiyiDrawer/workspace/civitai_meta'
    local_dir='E:\\sinri\\TaiyiDrawer\\workspace\\civitai_meta'
    cc.fetch_all_to_local(321, local_dir)
