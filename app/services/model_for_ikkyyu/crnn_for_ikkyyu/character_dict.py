class character_dict(object):
    MARKET_ONEHOT = {'1': 0, '0': 1, '8': 2, '5': 3, '3': 4, '-': 5, '2': 6, '=': 7, '6': 8, '×': 9, '7': 10, '÷': 11,
                     '4': 12, '9': 13, '+': 14, '捡': 15, '起': 16, '来': 17, '吧': 18, '错': 19, '题': 20, '本': 21, '把': 22,
                     '掉': 23, '落': 24, '的': 25, '(': 26, ')': 27, '*': 28, '口': 29, '算': 30, '练': 31, '习': 32, '>': 33,
                     '笔': 34, '闯': 35, '关': 36, '家': 37, '长': 38, '评': 39, '分': 40, ':': 41, '用': 42, '时': 43, '!': 44,
                     '第': 45, '天': 46, '月': 47, '日': 48, '乘': 49, '号': 50, '竖': 51, '式': 52, '脱': 53, '计': 54, '获': 55,
                     '得': 56, '收': 57, '了': 58, '巧': 59, '个': 60, '称': 61, '冲': 62, '刺': 63, '回': 64, '准': 65, '确': 66,
                     '率': 67, '开': 68, '始': 69, '基': 70, '础': 71, '过': 72, '小': 73, '朋': 74, '友': 75, ',': 76, '打': 77,
                     '上': 78, '对': 79, '勾': 80, '文': 81, '具': 82, '盒': 83, '才': 84, '可': 85, '以': 86, '购': 87, '物': 88,
                     '车': 89, '里': 90, '哦': 91, '你': 92, '几': 93, '呢': 94, '?': 95, '被': 96, '能': 97, '力': 98, '提': 99,
                     '高': 100, '@': 101, '初': 102, '级': 103, '银': 104, '员': 105, '中': 106, '钟': 107, '~': 108, '.': 109,
                     "'": 110, '未': 111, '借': 112, '位': 113, '有': 114, '棉': 115, '花': 116, '背': 117, '篓': 118, '优': 119,
                     '括': 120, '误': 121, '八': 122, '火': 123, '锅': 124, '菜': 125, '%': 126, '金': 127, '针': 128, '菇': 129,
                     '好': 130, '秒': 131, '父': 132, '运': 133, '符': 134, '变': 135, '化': 136, '前': 137, '是': 138, '减': 139,
                     '正': 140, '棒': 141, '哒': 142, '学': 143, '年': 144, '数': 145, '。': 146, '极': 147, '－': 148, '如': 149,
                     '拣': 150, 'd': 151, '巴': 152, 'o': 153, 'G': 154, '‘': 155, 'e': 156, 'N': 157, '￥': 158, '周': 159,
                     'n': 160, 's': 161, 'S': 162, '加': 163, '': -1}

    TOC_ONEHOT = {'○': 164, '子': 165, '兔': 166, '多': 167, '萝': 168, '两': 169, '共': 170, '一': 171, '少': 172, '卜': 173,
                  '拔': 174, '只': 175, 'P': 176, '展': 177, 'Ⅱ': 178, '拓': 179, '园': 180, '订': 181, '太': 182, '道': 183,
                  '每': 184, '}': 185, '{': 186, '|': 187, '看': 188, '寻': 189, '都': 190, '积': 191, '觅': 192, '致': 193,
                  '大': 194, '除': 195, '方': 196, '法': 197, '特': 198, '邻': 199, '殊': 200, '累': 201, '最': 202, '因': 203,
                  '短': 204, '例': 205, '质': 206, '细': 207, '便': 208, '相': 209, '利': 210, '、': 211, '互': 212, '公': 213,
                  '案': 214, 'J': 215, '篇': 216, '二': 217, '成': 218, '赛': 219, '附': 220, '三': 221, '玉': 222, '届': 223,
                  '属': 224, '林': 225, '节': 226, 'x': 227, '五': 228, '六': 229, '内': 230, '卡': 231, '生': 232, '出': 233,
                  '母': 234, '写': 235, '在': 236, '组': 237, '下': 238, '面': 239, '和': 240, '各': 241, '间': 242, 'X': 243,
                  '期': 244, '版': 245, '星': 246, '<': 247, '列': 248, '做': 249, '我': 250, '自': 251, '价': 252, 'B': 253,
                  '点': 254, '四': 255, '认': 256, '梁': 257, '辰': 258, '卷': 259, '步': 260, 't': 261, 'y': 262, 'h': 263,
                  'z': 264, 'a': 265, 'g': 266, 'i': 267, '午': 268, '再': 269, '测': 270, '目': 271, '答': 272, '实': 273,
                  '界': 274, '烛': 275, '灯': 276, '真': 277, '与': 278, '境': 279, '光': 280, '人': 281, '引': 282, '识': 283,
                  '知': 284, '导': 285, '到': 286, '明': 287, '复': 288, '混': 289, '合': 290, '·': 291, '还': 292, '行': 293,
                  '录': 294, '记': 295, '根': 296, '图': 297, '据': 298, '平': 299, '填': 300, '或': 301, '"': 302, '单': 303,
                  '元': 304, '整': 305, '课': 306, '理': 307, '刘': 308, '腾': 309, '战': 310, '挑': 311, '奥': 312, '绩': 313,
                  'u': 314, 'k': 315, '北': 316, '[': 317, '师': 318, ']': 319, '贴': 320, '完': 321, '签': 322, '字': 323,
                  '总': 324, '班': 325, '码': 326, '描': 327, '扫': 328, '维': 329, '油': 330, '快': 331, '谁': 332, '又': 333,
                  '专': 334, '江': 335, '苏': 336, '妈': 337, '不': 338, '试': 339, 'A': 340, '动': 341, '片': 342, '非': 343,
                  '常': 344, '超': 345, '老': 346, '语': 347, '直': 348, '接': 349, '进': 350, '观': 351, '体': 352, '察': 353,
                  '验': 354, '十': 355, '闰': 356, '鸡': 357, '今': 358, '爷': 359, '卖': 360, '王': 361, '集': 362, '\n': 363,
                  '猿': 364, '同': 365, '简': 366, '连': 367, '证': 368, '注': 369, '序': 370, '顺': 371, '定': 372, '性': 373,
                  '意': 374, '保': 375, '要': 376, '许': 377, '姓': 378, '名': 379, '英': 380, '灵': 381, '册': 382, '通': 383,
                  'r': 384, 'T': 385, '功': 386, '房': 387, '士': 388, '^': 389, 'Ⅰ': 390, '晚': 391, '训': 392, '机': 393,
                  '飞': 394, '继': 395, '很': 396, '努': 397, '续': 398, '主': 399, '阅': 400, '批': 401, '综': 402, 'R': 403,
                  '速': 404, '历': 405, '比': 406, '较': 407, '配': 408, '教': 409, '晓': 410, '全': 411, '示': 412, '表': 413,
                  '形': 414, '七': 415, '红': 416, '梅': 417, '张': 418, '汽': 419, '际': 420, '手': 421, '区': 422, '叶': 423,
                  '枫': 424, '易': 425, '程': 426, '置': 427, '信': 428, '窗': 429, '息': 430, '给': 431, '买': 432, '约': 433,
                  '叫': 434, '儿': 435, '童': 436, '餐': 437, '套': 438, '结': 439, '差': 440, '果': 441, '诀': 442, '句': 443,
                  '难': 444, '颗': 445, '豆': 446, '轻': 447, '松': 448, '百': 449, '解': 450, '舞': 451, '者': 452, 'l': 453,
                  '书': 454, '剩': 455, '跳': 456, '绳': 457, '冀': 458, '贵': 459, '恒': 460, '次': 461, '娃': 462, '魔': 463,
                  '会': 464, '满': 465, '备': 466, 'L': 467, '让': 468, '们': 469, '吗': 470, 'b': 471, 'q': 472, '并': 473,
                  '另': 474, '着': 475, '跟': 476, '也': 477, '新': 478, '升': 479, 'U': 480, 'O': 481, 'I': 482, 'K': 483,
                  'H': 484, 'E': 485, '页': 486, '见': 487, '某': 488, '这': 489, '值': 490, '夺': 491, '旗': 492, 'C': 493,
                  'Q': 494, '养': 495, 'p': 496, 'c': 497, 'm': 498, '由': 499, '建': 500, '创': 501, '阶': 502, '段': 503,
                  '走': 504, '议': 505, '早': 506, '改': 507, '堂': 508, '随': 509, '必': 510, '义': 511, '读': 512, '华': 513,
                  '等': 514, '腰': 515, '角': 516, '判': 517, '份': 518, '断': 519, '岁': 520, '乐': 521, '纸': 522, '克': 523,
                  '材': 524, '预': 525, '九': 526, '则': 527, '船': 528, '帆': 529, '右': 530, '气': 531, '说': 532, '指': 533,
                  '任': 534, '淀': 535, '海': 536, '爱': 537, '工': 538, '作': 539, '惜': 540, '么': 541, '那': 542, '辆': 543,
                  '按': 544, '求': 545, '补': 546, '充': 547, '外': 548, '钱': 549, '找': 550, '应': 551, '贫': 552, '山': 553,
                  '困': 554, '偏': 555, '思': 556, '庸': 557, '温': 558, '故': 559, '达': 560, '冉': 561, '排': 562, '从': 563,
                  '休': 564, '爸': 565, '消': 566, '防': 567, '孩': 568, '淘': 569, '熊': 570, '宝': 571, '店': 572, '铺': 573,
                  '型': 574, '般': 575, '类': 576, '选': 577, '团': 578, '就': 579, '量': 580, '命': 581, '制': 582, '重': 583,
                  '强': 584, '先': 585, '后': 586, '亲': 587, '辅': 588, '须': 589, '使': 590, '美': 591, '•': 592, '慧': 593,
                  '智': 594, '近': 595, '勿': 596, '听': 597, '标': 598, '封': 599, '含': 600, '想': 601, '照': 602, '样': 603,
                  '退': 604, '己': 605, '窝': 606, '于': 607, '善': 608, '伟': 609, '袋': 610, '米': 611, '安': 612, '装': 613,
                  '范': 614, '围': 615, '之': 616, ';': 617, '增': 618, '呦': 619, '检': 620, '末': 621, '香': 622, '秦': 623,
                  '微': 624, 'M': 625, 'w': 626, '众': 627, '更': 628, '万': 629, '《': 630, '》': 631, '土': 632, '既': 633,
                  '舍': 634, '镂': 635, '石': 636, '锲': 637, '而': 638, '它': 639, '零': 640, '荚': 641, '为': 642, '适': 643,
                  '航': 644, '余': 645, '景': 646, '空': 647, '梳': 648, '镜': 649, '妆': 650, '修': 651, '营': 652, '业': 653,
                  '请': 654, '圈': 655, '哪': 656, '倍': 657, '存': 658, '系': 659, '捐': 660, '些': 661, '怎': 662, '横': 663,
                  '捕': 664, '猎': 665, '网': 666, '忆': 667, '刻': 668, '捺': 669, '“': 670, '左': 671, '谦': 672, '边': 673,
                  '”': 674, '树': 675, '栽': 676, '盘': 677, '现': 678, '发': 679, '李': 680, '龙': 681, '逸': 682, '均': 683,
                  '□': 684, '画': 685, '④': 686, '⑥': 687, '⑨': 688, '⑤': 689, '⑦': 690, '⑩': 691, '器': 692, '琢': 693,
                  '律': 694, '规': 695, '视': 696, 'j': 697, '什': 698, '线': 699, '棵': 700, '放': 701, '舒': 702, '木': 703,
                  '撇': 704, '固': 705, '巩': 706, '梨': 707, '仔': 708, '笑': 709, '郎': 710, '坚': 711, '持': 712, '终': 713,
                  '懈': 714, '③': 715, '②': 716, '①': 717, '考': 718, '底': 719, '代': 720, '远': 721, '醒': 722, '苹': 723,
                  '旁': 724, '；': 725, '卧': 726, '钩': 727, '折': 728}

    TOC_ONEHOT_7 = {'厘': 729, '宽': 730, '当': 731, '吃': 732, '害': 733, '虫': 734, '经': 735, '跑': 736, '已': 737, '估': 738,
                    '≈': 739, '毫': 740, '件': 741, '丽': 742, '粒': 743, '锐': 744, '商': 745, '千': 746, '科': 747, '技': 748,
                    '环': 749, '身': 750, '暑': 751, '假': 752, '档': 753, '朵': 754, '敏': 755, '草': 756, '活': 757, '向': 758,
                    '斑': 759, '马': 760, '虎': 761, '及': 762, '析': 763, '钢': 764, '越': 765, '品': 766, '条': 767, '冬': 768,
                    '摘': 769, '楼': 770, '奶': 771, '牛': 772, '需': 773, '层': 774, '柳': 775, '杨': 776, '织': 777, '支': 778,
                    '鹅': 779, '鸭': 780, '室': 781, '莉': 782, '尾': 783, '圆': 784, '珠': 785, '铅': 786, '贺': 787, '调': 788,
                    '问': 789, '决': 790, '带': 791, '玩': 792, '够': 793, '双': 794, '牡': 795, '芍': 796, '种': 797, '丹': 798,
                    '药': 799, '头': 800, '男': 801, '女': 802, '鹤': 803, '鲜': 804, '盆': 805, '包': 806, '赵': 807, '晨': 808,
                    '宇': 809, '凯': 810, '番': 811, '茄': 812, '门': 813, '株': 814, '击': 815, '箱': 816, '宫': 817, '探': 818,
                    '秘': 819, '鸟': 820, '台': 821, '事': 822, '话': 823, '付': 824, '△': 825, '柿': 826, '瓜': 827, '黄': 828,
                    '西': 829, '®': 830, '肥': 831, '猫': 832, '鱼': 833, '钝': 834, '丝': 835, '铁': 836, '史': 837, '站': 838,
                    '枚': 839, '邮': 840, '票': 841, '洗': 842, '衣': 843, '龄': 844, '浩': 845, '鞋': 846, '袜': 847, '拖': 848,
                    '拿': 849, '校': 850, '吨': 851, '池': 852, '原': 853, '塘': 854, '眼': 855, '涂': 856, '色': 857, '颜': 858,
                    '浪': 859, '路': 860, '去': 861, '踢': 862, '格': 863, '厚': 864, '块': 865, '游': 866, '春': 867, '顶': 868,
                    '帽': 869, '售': 870, '销': 871, '参': 872, '操': 873, '坐': 874, '租': 875, '划': 876, '筷': 877, '丁': 878,
                    '隔': 879, '幼': 880, '育': 881, '馆': 882, '瓶': 883, '立': 884, '启': 885, '唱': 886, '狗': 887, '猪': 888,
                    '球': 889, '篮': 890, '硬': 891, '币': 892, '务': 893, '所': 894, '统': 895, '灾': 896, '况': 897, '情': 898,
                    '其': 899, '坦': 900, '别': 901, '莓': 902, '换': 903, '服': 904, '禽': 905, '乖': 906, '羊': 907, '巾': 908,
                    '世': 909, '客': 910, '博': 911, '他': 912, '丈': 913, '户': 914, '瓦': 915, '电': 916, '阿': 917, '姨': 918,
                    '部': 919, '狮': 920, '饼': 921, '干': 922, '糕': 923, '送': 924, '蛋': 925, '将': 926, '▪': 927, '洁': 928,
                    '社': 929, '徽': 930, '推': 931, '座': 932, '且': 933, '橡': 934, '皮': 935, '桃': 936, '取': 937, '堆': 938,
                    '沙': 939, '场': 940, '农': 941, '裤': 942, '匡': 943, '厂': 944, '冰': 945, '白': 946, '’': 947, '候': 948,
                    '框': 949, '粉': 950, '润': 951, '城': 952, '蕉': 953, '术': 954, '践': 955, '半': 956, '无': 957, '否': 958,
                    '拨': 959, '劫': 960, '至': 961, '住': 962, '彩': 963, '没': 964, '齐': 965, '射': 966, '箭': 967, '靶': 968,
                    '度': 969, '择': 970, '昨': 971, '桌': 972, '床': 973, '队': 974, '摆': 975, '禾': 976, '葡': 977, '萄': 978,
                    '鑫': 979, '陈': 980, '该': 981, '绿': 982, '≥': 983, '异': 984, '芳': 985, '良': 986, '饭': 987, '蜡': 988,
                    '娄': 989, '核': 990, '军': 991, '讲': 992, '黑': 993, '摩': 994, '查': 995, '失': 996, '&': 997}

    TOC_ONEHOT_7F = {'渝': 998, '²': 999, 'π': 1000, '循': 1001, '入': 1002, '∟': 1003, '秋': 1004, '季': 1005, '艾': 1006,
                     '设': 1007, '迪': 1008, '莫': 1009, '尔': 1010, '哈': 1011, '艺': 1012, '斯': 1013, '辍': 1014, '崇': 1015,
                     '鳖': 1016, '跬': 1017, '跛': 1018, '丘': 1019, '象': 1020, '严': 1021, '抽': 1022, '曼': 1023, '穷': 1024,
                     '赫': 1025, '丨': 1026, '市': 1027, '言': 1028, '粹': 1029, '诗': 1030, '逻': 1031, '辑': 1032, '纯': 1033,
                     '演': 1034, '锻': 1035, '挥': 1036, '精': 1037, '独': 1038, '炼': 1039, '神': 1040, '健': 1041, '康': 1042,
                     '荀': 1043, '毕': 1044, '拉': 1045, '治': 1046, '宙': 1047, '哥': 1048, '货': 1049, '德': 1050, '造': 1051,
                     '促': 1052, '秩': 1053, '亚': 1054, '丙': 1055, '乙': 1056, '甲': 1057, '…': 1058, '缺': 1059, '陷': 1060,
                     '势': 1061, '莱': 1062, '冠': 1063, '烂': 1064, '灿': 1065, '皇': 1066, '但': 1067, '拥': 1068, '罗': 1069,
                     '地': 1070, '素': 1071, '扩': 1072, '域': 1073, '领': 1074, '穿': 1075, '嵩': 1076, '束': 1077, '感': 1078,
                     '杵': 1079, '磨': 1080, '觉': 1081, '夫': 1082, '何': 1083, '深': 1084, '秀': 1085, '腿': 1086, '笼': 1087,
                     '薇': 1088, '↓': 1089, '论': 1090, '然': 1091, '免': 1092, '避': 1093, '蛮': 1094, '菠': 1095, '培': 1096,
                     '切': 1097, '柏': 1098, '帝': 1099, '述': 1100, '概': 1101, '念': 1102, '钥': 1103, '匙': 1104, '雅': 1105,
                     '显': 1106, '音': 1107, '助': 1108, '呈': 1109, '献': 1110, '辉': 1111, '煌': 1112, '庚': 1113, '鳌': 1114,
                     '迈': 1115, '怕': 1116, '倒': 1117, '魏': 1118, '↑': 1119, '匠': 1120, '心': 1121, '东': 1122, '违': 1123,
                     '髓': 1124, '彻': 1125, '朽': 1126, '硅': 1127, '青': 1128, '索': 1129}
                     
    TOC_ONEHOT_8 =  {'丶': 1130, '熟': 1131, '炳': 1132, '麒': 1133, '姚': 1134, '筐': 1135, '翔': 1136, '沪': 1137, '研': 1138,
                      'f': 1139, '官': 1140, '阳': 1141, '板': 1142, '饰': 1143, '枢': 1144, '焊': 1145, '捆': 1146, '婷': 1147,
                      '处': 1148, '帮': 1149, '她': 1150, '哇': 1151, '架': 1152, '食': 1153, '栏': 1154, '争': 1155, '伍': 1156,
                      '疑': 1157, '往': 1158, '诊': 1159, '⑧': 1160, '胖': 1161, '典': 1162, '钉': 1163, '杆': 1164, '政': 1165,
                      '朱': 1166, '静': 1167, '邱': 1168, '纠': 1169, '睿': 1170, '熙': 1171, '医': 1172, '疗': 1173, '构': 1174,
                      '院': 1175, '勺': 1176, '搭': 1177, '雪': 1178, '蛙': 1179, '拼': 1180, '替': 1181, '淡': 1182, '菊': 1183,
                      '紫': 1184, '睡': 1185, '洞': 1186, '离': 1187, '词': 1188, '桐': 1189, '乡': 1190, '尖': 1191, '寒': 1192,
                      '丛': 1193, '风': 1194, '迎': 1195, '蟹': 1196, '螃': 1197, '`': 1198, '贝': 1199, 'D': 1200, '牌': 1201,
                      '赢': 1202, '资': 1203, '搜': 1204, '费': 1205, '料': 1206, '仿': 1207, '捉': 1208, '遍': 1209, '轩': 1210,
                      '懿': 1211, '像': 1212, '欢': 1213, '喜': 1214, '沽': 1215, '添': 1216, '款': 1217, '勤': 1218, '滴': 1219,
                      '剪': 1220, '赶': 1221, '宣': 1222, '柴': 1223, '瓣': 1224, '皓': 1225, '南': 1226, '毛': 1227, '胡': 1228,
                      '扣': 1229, '孙': 1230, '【': 1231, '受': 1232, '益': 1233, '敲': 1234, '清': 1235, '态': 1236, 'F': 1237,
                      '刚': 1238, '项': 1239, '鹰': 1240, '戏': 1241, '甫': 1242, '杂': 1243}
    
    TOC_ONEHOT_9 = {'V': 1254, '锥': 1251, '∶': 1252, '柱': 1249, 'v': 1248, '≠': 1250, '∠': 1244, '％': 1253, '亿': 1256, '顷': 1246, '枝': 1255, '侧': 1247, '～': 1258, '杯': 1257, '°': 1245}
