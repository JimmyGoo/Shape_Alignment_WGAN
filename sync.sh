rsync -av --progress --rsh='ssh' --exclude ".git/" --exclude ".DS_Store" --exclude "model/" --exclude "vis/" --exclude "temp.py" --exclude "create_tfrecord_old.py" --exclude "data/" --exclude "*.pyc" --exclude "log/" -r ./ yangqingzhu@166.111.7.190:~/jimmygoo/3D_Alignment_WGAN/
