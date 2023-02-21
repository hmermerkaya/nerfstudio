# Web Deployment-Nerfstudio Overview
This is the documentation of installation of  nerf-studio and its web-deployment. By this, you can run nerfstudio, the bridge server and viewer [ReactJS](https://reactjs.org/) app on the cloud and view it remotely. 

## Steps:
 Launch an Ubuntu 20.04 AWS instance having NVIDIA video card with CUDA compute capability of at least 7.0. (e.g. g5.xlarge). Make sure that port numbers 80, 4000-4010 
 and 7000-7010 on the cloud are allowed.

 Install Miniconda with python 3.8


```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh
bash Miniconda3-py38_23.1.0-1-Linux-x86_64.sh 
    
```
exit the terminal and relogin.

Install CUDA 11.3 with latest compatible nvidia-driver using following commands and run “nvidia-smi” to make sure nvidia-driver is installed.
```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda-11-3
sudo reboot
nvidia-smi 
```
Set up “Nerfstudio” environment. 

```shell
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
python -m pip install --upgrade pip
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Clone the repository of “Nerfstudio”  and install it.

```shell
git clone https://github.com/hmermerkaya/nerfstudio.git
cd nerfstudio
pip install --upgrade pip setuptools
pip install -e .
```
Run nerfstudio training on a default sample
```shell
ns-download-data nerfstudio –capture-name=poster
ns-train nerfacto --data data/nerfstudio/poster
```

Let it run for at least 2000 iteration to get a descent trained checkpoint.

Install viewer react app and run locally.
```shell       
cd nerfstudio/viewer/app
sudo apt-get install npm
sudo npm install --global yarn
curl -sL https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.0/install.sh -o install_nvm.sh
bash install_nvm.sh
```

exit current terminal and open new one.
```shell
nvm install 17.8.0
yarn install
yarn start
```

Press CTRL+C to exit.

Install pm2 (PM2 is a Production Process Manager for Node.js applications) and do settings about it.
```shell
sudo npm i -g pm2@latest
cd nerfstudio/nerfstudio/viewer/app
pm2 start yarn --name "nerf viewer" – start
pm2 save
pm2 startup systemd
sudo env PATH=$PATH:/home/ubuntu/.nvm/versions/node/v17.8.0/bin /usr/local/lib/node_modules/pm2/bin/pm2 startup systemd -u ubuntu --hp /home/ubuntu
```

Install nginx and do settings.
```shell
sudo apt install nginx -y
cd /etc/nginx/sites-available/
```

Create a new file named like ‘nerf-viewer’ with the content below.
```
server {
    	    listen 80;
    	    listen [::]:80;

    server_name ip_address;
    access_log /var/log/nginx/reat-tutorial.com.access.log;
    error_log /var/log/nginx/reat-tutorial.com.error.log;
    location / {
            proxy_pass http://127.0.0.1:4000;
            client_max_body_size 50m;
            client_body_buffer_size 16k;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_cache_bypass $http_upgrade;
    }
}
```

ip_address above must be replaced by  server name  or IP address of AWS instance.
```shell
sudo ln -s /etc/nginx/sites-available/nerf-viewer  /etc/nginx/sites-enabled/nerf-viewer
sudo systemctl restart nginx
```
Run the trained checkpoint.
```shell
cd nerfstudio
conda activate nerfstudio
scripts/render_view.py nerfacto-render –load-dir=/path/to/config.yml/of/trainedcheckpoint --viewer.websocket-port=7007 –viewer.start_train=False
```
View the REACTJS viewer app at local from the link below.
```
ip_address/?websocket_url=ws://ip_adress:7007
```
where ip_address is the IP address of  AWS instance.
