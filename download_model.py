#验证 ModelScope token
from modelscope.hub.api import HubApi
api = HubApi()
api.login('ms-f54736da-0bd3-4307-9768-3c35f40ba09b')

model_name = 'hurrylu/Qwen3-0.6B-ADDR64V1'
local_model_path = './local-model'

#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download(
    model_id=model_name,
    cache_dir=local_model_path,
    revision="master"
)