### 一：安装常见知识点

使用conda升级Python及第三方库

```Python
conda update conda
```

conda是Anaconda的管理工具，在更新Anaconda之前，需要对conda工具本身进行升级。各个系统都适用。更新完毕后，运行一下代码更新Anaconda：

```Python
conda update anaconda
```

更新完成后，运行一下更新Python

```Python
conda update python
```

检查一次python版本

```Python
python --version

```

 使用conda对第三方库进行更新

```Python
conda update --all
```

有些库不会出现在conda的库中，有时候也会使用pip进行库的更新

```Python
python -m pip install --upgrade pip
```

有时候更新较慢，需要使用国内的镜像源

### 二：常用国内镜像源地址汇总

|镜像源|镜像源地址|
|-|-|
|阿里云|[http://mirrors.aliyun.com/pypi/simple/](http://mirrors.aliyun.com/pypi/simple/)|
|中国科技大学|[https://pypi.mirrors.ustc.edu.cn/simple/](https://pypi.mirrors.ustc.edu.cn/simple/)|
|豆瓣(douban)|[http://pypi.douban.com/simple/](http://pypi.douban.com/simple/)|
|清华大学|[https://pypi.tuna.tsinghua.edu.cn/simple/](https://pypi.tuna.tsinghua.edu.cn/simple/)|
|中国科学技术大学|[http://pypi.mirrors.ustc.edu.cn/simple/](http://pypi.mirrors.ustc.edu.cn/simple/)|


使用方法：以Python环境下pip安装[keras](https://so.csdn.net/so/search?q=keras&spm=1001.2101.3001.7020)为例

```Python
pip install keras==2.3.1 -i http://pypi.douban.com/simple/
//2.3.1为所需版本号，尾部url为对应镜像源地址

```

### 三：镜像源

3.1永久添加镜像源

```Python
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/

conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2

conda config --set show_channel_urls yes

```

3.2以下命令验证一下是不是配置成功

```Python
conda config --show channels

```

3.3之后再进行conda更新

```Python
conda update conda

```

### 四：Pytroch安装及环境配置

**进入下面的地址**[默认装好cuda和cudnn]

```Python
https://pytorch.org/get-started/locally/
```

#### 4.1在Win10系统下配置PyTorch

![](https://secure2.wostatic.cn/static/3rJCdxLNmsTEos5BgxzYtm/image.png?auth_key=1669193354-oEYPLA5qQ3zRHY1zr7xK5X-0-f1b471b5446f38731dfe8dde23aa7a5a)

#### 4.2在Ubuntu系统下配置PyTorch

![](https://secure2.wostatic.cn/static/cAhiframTyAzMQxVEqFFBk/image.png?auth_key=1669193401-puJsCkDmAEDEC4keWXsg3Q-0-f6aaa64b55bf1c9073ac2b0a973d9e41)

#### 4.3在Mac系统下配置PyTorch

![](https://secure2.wostatic.cn/static/8DNMVqLSMjZgSM3SR7MA2X/image.png?auth_key=1669193442-k2RaiS5SXckE1osoNMAiK3-0-91b5aa18abfddd2e17c844de7362506f)
