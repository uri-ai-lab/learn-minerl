# learn-minerl
AI Lab repo for code samples and documentation to help users of [MineRL](https://minerl.io), a large scale dataset for machine learning experiments based on Minecraft user interactions.

###### MineRL Installation

Operating System: Ubuntu 18.04.5 LTS
Python Version:   Python 3.6.13 :: Anaconda, Inc.

**Install conda**

**Install JDK 1.8 (Must set env variable java after installation)**

sudo add-apt-repository ppa:openjdk-r/ppa
sudo apt-get update
sudo apt-get install openjdk-8-jdk

**Create python evironment for MineRL**

conda create -n mineRL-py36 python=3.6

**Activate conda MineRL environment**

conda activate mineRL-py36

**Install MineRL and dependecies**

pip3 install --upgrade minerl
conda install tqdm
pip3 install scikit-learn
pip3 install pyglet==1.5.11

**Manually editting files**

https://github.com/minerllabs/minerl/issues/450#issuecomment-777009360

https://github.com/minerllabs/minerl/commit/548116461c7213caf23029c7651086fea22d21e9