## instal anaconda:
you need to download the .deb file from this link  
[download anaconda ](https://www.anaconda.com/products/individual#linux)

then you can install this file with this command:  
```
  sudo bash ./your_file_name
```
## Installing pip for Python 3  
1. Start by updating the package list using the following command:
`sudo apt update`
2. Use the following command to install pip for Python 3:  
`sudo apt install python3-pip`  
The command above will also install all the dependencies required for building Python modules.  
3. Once the installation is complete, verify the installation by checking the pip version:
`pip3 --version`  

## install Keras 

conda command doesn't work with sudo. To fix this error we need to give permission to anaconda folder by changing ownership.  
`sudo chown -R myuser /home/myuser/anaconda3`  
then we can install conda with this command:
`conda install keras`  

install tensorflow:  
# Requires the latest pip
`pip install --upgrade pip`

Current stable release for CPU and GPU:  
`pip install tensorflow`
