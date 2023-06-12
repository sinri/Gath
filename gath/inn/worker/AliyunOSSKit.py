# -*- coding: utf-8 -*-
from typing import Optional

import oss2

from gath import env


class AliyunOSSKit:
    def __init__(self):
        """
        阿里云账号AccessKey拥有所有API的访问权限，风险很高。强烈建议您创建并使用RAM用户进行API访问或日常运维，请登录RAM控制台创建RAM用户。
        """
        self.__auth = oss2.Auth(env.inn_oss_ak_id, env.inn_oss_ak_secret)

    def upload_local_file(self, local_path: str, oss_path: str, bucket: Optional[str] = None,
                          endpoint: Optional[str] = None):
        """
        上传文件到OSS。
        <yourObjectName>由包含文件后缀，不包含Bucket名称组成的Object完整路径，例如abc/efg/123.jpg。
        <yourLocalFile>由本地文件路径加文件名包括后缀组成，例如/users/local/myfile.txt。
        Endpoint以杭州为例，其它Region请按实际情况填写。
        """
        if bucket is None:
            bucket = env.inn_oss_bucket
        if endpoint is None:
            endpoint = env.inn_oss_endpoint
        bucket = oss2.Bucket(self.__auth, endpoint, bucket)

        return bucket.put_object_from_file(oss_path, local_path)
