from setuptools import setup, find_packages

setup(
    name="ftools",
    version="1.0",
    author="fangyaozheng",
    author_email="fyz@mail.nankai.edu.cm",
    description="Some common tools for daily development.",

    # 项目主页
    url="http://ftools.dev/", 

    # 你要安装的包，通过 setuptools.find_packages 找到当前目录下有哪些包
    packages=find_packages()
)