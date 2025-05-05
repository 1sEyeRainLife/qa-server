test_questions = [
    {
        "question": "水的沸点是多少摄氏度？",
        "answer": "100摄氏度",
        "category": "科学",
        "relevant_docs": ["physics_101", "water_properties"]
    },
    {
        "question": "《红楼梦》的作者是谁？",
        "answer": "曹雪芹",
        "category": "文学",
        "relevant_docs": ["chinese_literature", "hongloumeng"]
    },
    {
        "question": "Python是哪一年发布的？",
        "answer": "1991年",
        "category": "计算机",
        "relevant_docs": ["python_history", "programming_languages"]
    },
    {
        "question": "人类有多少对染色体？",
        "answer": "23对",
        "category": "生物学",
        "relevant_docs": ["human_genetics", "biology_basics"]
    },
    {
        "question": "第一次世界大战爆发于哪一年？",
        "answer": "1914年",
        "category": "历史",
        "relevant_docs": ["ww1_history", "20th_century_events"]
    },
    {
        "question": "太阳系中最大的行星是哪个？",
        "answer": "木星",
        "category": "天文",
        "relevant_docs": ["solar_system", "planets_info"]
    },
    {
        "question": "iPhone的第一个版本是哪一年发布的？",
        "answer": "2007年",
        "category": "科技",
        "relevant_docs": ["apple_history", "smartphone_evolution"]
    },
    {
        "question": "光的传播速度是多少？",
        "answer": "约每秒299,792公里",
        "category": "物理",
        "relevant_docs": ["physics_constants", "light_properties"]
    },
    {
        "question": "中国最长的河流是哪条？",
        "answer": "长江",
        "category": "地理",
        "relevant_docs": ["china_geography", "world_rivers"]
    },
    {
        "question": "DNA的全称是什么？",
        "answer": "脱氧核糖核酸",
        "category": "生物学",
        "relevant_docs": ["genetics_basics", "dna_structure"]
    },
    {
        "question": "《蒙娜丽莎》的作者是谁？",
        "answer": "达·芬奇",
        "category": "艺术",
        "relevant_docs": ["renaissance_art", "leonardo_da_vinci"]
    },
    {
        "question": "亚马逊河位于哪个洲？",
        "answer": "南美洲",
        "category": "地理",
        "relevant_docs": ["south_america_geo", "major_rivers"]
    },
    {
        "question": "Windows操作系统的开发者是哪家公司？",
        "answer": "微软",
        "category": "计算机",
        "relevant_docs": ["microsoft_history", "os_development"]
    },
    {
        "question": "元素周期表中第一个元素是什么？",
        "answer": "氢",
        "category": "化学",
        "relevant_docs": ["periodic_table", "chemical_elements"]
    },
    {
        "question": "联合国总部位于哪个城市？",
        "answer": "纽约",
        "category": "政治",
        "relevant_docs": ["un_organization", "international_institutions"]
    },
    {
        "question": "正常成年人的体温范围是多少？",
        "answer": "36.5-37.5摄氏度",
        "category": "医学",
        "relevant_docs": ["human_physiology", "medical_basics"]
    },
    {
        "question": "《哈利·波特》系列小说的作者是谁？",
        "answer": "J.K.罗琳",
        "category": "文学",
        "relevant_docs": ["british_literature", "harry_potter_series"]
    },
    {
        "question": "地球的天然卫星是什么？",
        "answer": "月球",
        "category": "天文",
        "relevant_docs": ["earth_moon", "solar_system_bodies"]
    },
    {
        "question": "诺贝尔奖共有几个奖项类别？",
        "answer": "6个",
        "category": "文化",
        "relevant_docs": ["nobel_prizes", "international_awards"]
    },
    {
        "question": "中国的首都是哪里？",
        "answer": "北京",
        "category": "地理",
        "relevant_docs": ["china_facts", "world_capitals"]
    },
    {
        "question": "互联网上最常用的协议是什么？",
        "answer": "HTTP/HTTPS",
        "category": "计算机",
        "relevant_docs": ["internet_protocols", "web_technology"]
    },
    {
        "question": "哺乳动物的特征是什么？",
        "answer": "用肺呼吸、胎生、哺乳",
        "category": "生物学",
        "relevant_docs": ["animal_classification", "mammal_characteristics"]
    },
    {
        "question": "珠穆朗玛峰的高度是多少米？",
        "answer": "8848.86米",
        "category": "地理",
        "relevant_docs": ["highest_mountains", "geography_records"]
    },
    {
        "question": "Photoshop是由哪家公司开发的？",
        "answer": "Adobe",
        "category": "科技",
        "relevant_docs": ["adobe_products", "graphic_design"]
    },
    {
        "question": "莎士比亚的四大悲剧包括哪些？",
        "answer": "《哈姆雷特》、《奥赛罗》、《李尔王》、《麦克白》",
        "category": "文学",
        "relevant_docs": ["shakespeare_works", "english_literature"]
    },
    {
        "question": "电的发明者是谁？",
        "answer": "电不是被发明的，但本杰明·富兰克林进行了重要研究",
        "category": "科学",
        "relevant_docs": ["electricity_history", "famous_scientists"]
    },
    {
        "question": "中国有多少个省级行政区？",
        "answer": "34个",
        "category": "地理",
        "relevant_docs": ["china_administrative", "country_divisions"]
    },
    {
        "question": "HIV病毒主要攻击人体的什么系统？",
        "answer": "免疫系统",
        "category": "医学",
        "relevant_docs": ["hiv_aids", "human_immunology"]
    },
    {
        "question": "马拉松比赛的标准距离是多少公里？",
        "answer": "42.195公里",
        "category": "体育",
        "relevant_docs": ["marathon_history", "sport_events"]
    },
    {
        "question": "谷歌公司成立于哪一年？",
        "answer": "1998年",
        "category": "科技",
        "relevant_docs": ["google_history", "tech_companies"]
    },
    {
        "question": "世界上最长的城墙是什么？",
        "answer": "中国的万里长城",
        "category": "历史",
        "relevant_docs": ["great_wall_china", "world_heritage_sites"]
    },
    {
        "question": "牛顿第一定律又称为什么？",
        "answer": "惯性定律",
        "category": "物理",
        "relevant_docs": ["newton_laws", "physics_basics"]
    },
    {
        "question": "《孙子兵法》的作者是谁？",
        "answer": "孙武",
        "category": "军事",
        "relevant_docs": ["chinese_military", "ancient_strategies"]
    },
    {
        "question": "地球上最大的洲是哪个？",
        "answer": "亚洲",
        "category": "地理",
        "relevant_docs": ["world_continents", "geography_facts"]
    },
    {
        "question": "Facebook的创始人是谁？",
        "answer": "马克·扎克伯格",
        "category": "科技",
        "relevant_docs": ["social_media_history", "tech_entrepreneurs"]
    },
    {
        "question": "水的化学式是什么？",
        "answer": "H₂O",
        "category": "化学",
        "relevant_docs": ["water_chemistry", "chemical_formulas"]
    },
    {
        "question": "《义勇军进行曲》是哪个国家的国歌？",
        "answer": "中国",
        "category": "文化",
        "relevant_docs": ["china_national_symbols", "world_anthems"]
    },
    {
        "question": "国际象棋共有多少个棋子？",
        "answer": "32个",
        "category": "游戏",
        "relevant_docs": ["chess_rules", "board_games"]
    },
    {
        "question": "爱因斯坦最著名的理论是什么？",
        "answer": "相对论",
        "category": "物理",
        "relevant_docs": ["einstein_theories", "modern_physics"]
    },
    {
        "question": "中国抗日战争持续了多少年？",
        "answer": "14年（1931-1945）",
        "category": "历史",
        "relevant_docs": ["sino_japanese_war", "ww2_history"]
    },
    {
        "question": "计算机的基本存储单位是什么？",
        "answer": "字节(Byte)",
        "category": "计算机",
        "relevant_docs": ["computer_storage", "it_basics"]
    },
    {
        "question": "世界上最深的海洋是哪个？",
        "answer": "太平洋",
        "category": "地理",
        "relevant_docs": ["oceanography", "earth_oceans"]
    },
    {
        "question": "《西游记》中唐僧的原型是谁？",
        "answer": "玄奘",
        "category": "文学",
        "relevant_docs": ["chinese_classics", "buddhist_history"]
    },
    {
        "question": "正常大气压下，水的冰点是多少摄氏度？",
        "answer": "0摄氏度",
        "category": "物理",
        "relevant_docs": ["water_properties", "phase_transitions"]
    },
    {
        "question": "奥运会几年举办一次？",
        "answer": "4年",
        "category": "体育",
        "relevant_docs": ["olympic_games", "international_sports"]
    },
    {
        "question": "中国的四大发明是什么？",
        "answer": "造纸术、印刷术、指南针、火药",
        "category": "历史",
        "relevant_docs": ["chinese_inventions", "ancient_technology"]
    },
    {
        "question": "植物通过什么过程制造氧气？",
        "answer": "光合作用",
        "category": "生物学",
        "relevant_docs": ["plant_physiology", "photosynthesis"]
    },
    {
        "question": "Windows 10发布于哪一年？",
        "answer": "2015年",
        "category": "计算机",
        "relevant_docs": ["windows_versions", "microsoft_os"]
    },
    {
        "question": "地球上最长的经线是多少度？",
        "answer": "0度（本初子午线）",
        "category": "地理",
        "relevant_docs": ["longitude_latitude", "earth_grid"]
    },
    {
        "question": "《战争与和平》的作者是谁？",
        "answer": "列夫·托尔斯泰",
        "category": "文学",
        "relevant_docs": ["russian_literature", "tolstoy_works"]
    },
    {
        "question": "人体最大的器官是什么？",
        "answer": "皮肤",
        "category": "医学",
        "relevant_docs": ["human_anatomy", "organ_systems"]
    }
]