# Automatically generate couplets
 使用seq2seq模型和attention机制自动生成对联 \
 目前已完成 \
 数据集链接： \
 https://disk.pku.edu.cn:443/link/4FF3DF1D5A06450D9F78935BE4E48EDA \
有效期限：2026-06-03 23:59 \

- web最终版：利用已经训练好的模型实现了网页展示，使用Vue和Flask进行前后端交互

  - python main.py即可打开网页（注意您的python除了有必备的torch、tf等库外，还要安装flask：`sudo pip install flask`）
  - 由于模型训练是用cuda跑的，所以调用model我们默认需要cuda，加入您没有cuda，不用着急，修改一下main.py里的：

- 对联建模版：使用seq2seq模型和attention机制进行模型训练，数据集链接如上
