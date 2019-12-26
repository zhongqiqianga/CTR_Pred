"""
配置文件，包含训练文件的目录，以及连续和离散值特征列表
"""
# set the path-to-files
TRAIN_FILE = "/cluster/workspace/zhongqiqiang/data/raw_data.csv"
TEST_FILE = "/cluster/workspace/zhongqiqiang/data/raw_data.csv"

SUB_DIR = "../output"

NUM_SPLITS = 2
RANDOM_SEED = 2019
# 不知道abtest是连续值还是离散值，item_rank
# 删除candidate_click_cnt
# types of columns of the dataset dataframe
CATEGORICAL_COLS = [
     'action_show','action_preshow' ,'predict_model_id', 'remotePort','contextExist','userId', 'logType','url', 'clientToken', 'lib', 'serviceName',
      'channel', 'pubIp', 'businessId', 'sceneId','libVersion', 'sceneName', 'businessToken', 'itemSetId'
     , 'position', 'category_list', 'noun_pros',
     'is_recommend', 'tag','id', 'item_progress_status', 'word_cnt', 'tag_url', 'title', 'category_v1', 'content',
    'source', 'cover_url_size','type', 'es_calculated', 'item_rank'
,'business_id', 'cover_url', 'nouns', 'item_id', 'es_pushed', 'qs', 'title_parent_id','publisher_id','location','category_id','user_interest_itemId',
    'user_interest_channel','abtest','ip','user_interest_tag','user_interest_category_v1'
]
IGNORE_COLS=[]
TIMES_COLS = ['action_detailPageShow_actionTime', 'action_preshow_actionTime', 'action_show_actionTime', 'actionTime', 'logWriteTimeHuman','logWriteTime', 'reqEndMillis', 'publish_time', 'utime', 'ctime']

NUMERIC_COLS = ['item_ctr','bonus','ctr','item_click_cnt','candidate_click_cnt',  'item_4pd_show_cnt', 'item_read_cnt','score','item_4pd_click_cnt',
                'item_show_cnt','cover_url_cnt','url_cnt'
]
