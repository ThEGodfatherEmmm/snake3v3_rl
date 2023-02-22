文件框架大体与源代码相同，环境也完全一致，训练好的agent在agent文件夹中，调用相应的submission即可。
rl_trainer是baseline的ddpg算法，进入文件夹运行main.py即可开始训练。训练结果和日志保存在model文件夹内，这点所有算法都一致。
evaluation_local.py文件将训练好的模型与random游走进行对比，适用性不强所以我没有调整参数，对不同的算法需要手动调整agent位置。
compare_local.py文件可以比较各个不同算法之间的效果，使用时添加参数--my_ai（默认SD3），--opponent（默认ddpg），--episode。