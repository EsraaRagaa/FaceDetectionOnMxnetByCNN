#########################################################################
# File Name: buildDatabase.sh
# Author: raymon-tian 
# Created Time: 2016年04月21日 星期四 21时24分23秒
#########################################################################
#!/bin/bash
python make_list.py data data/train --recursive=True
python im2rec.py data/train data
