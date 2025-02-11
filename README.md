# Image orientation detection
*App inspired by the work of [Deepanshu Thakur](https://github.com/Deepanshu2017/AlligatorOrCrocodile) and [Simon Willison](https://github.com/simonw/cougar-or-not)*

This webapp was created on behalf of the British Library [Endangered Archive Programme](https://eap.bl.uk/) to determine the orientation of images in their collection. The deep learning model is based on a 50-layer version of [resnet](https://arxiv.org/abs/1512.03385) and was trained on around 650 unique images from the EAP archives using the [fast.ai](https://docs.fast.ai/) library. Each image is duplicated into four representations, consisting of the application of 0, 90, 180 or 270 degrees of anti-clockwise rotation.

The 99MB pre-trained model is included in this package and is found within `resnet50_EAP_version6.pkl`

`imgapp.py` is a [Starlette](https://www.starlette.io/) API server which accepts image uploads or image URLs and runs them against the pre-calculated model.

A `Dockerfile` is included for convenience.

## Deployment
If deploying on a server, run the Starlette app on the uvicorn ASGI server behind an nginx reverse proxy

```
cd $APP_DIR
sudo apt-get install nginx
(sudo) cp nginx_config_file.conf /etc/nginx/sites-available/default
sudo ln -s /etc/nginx/sites-available/default /etc/nginx/sites-enabled/

```
*Before this step, edit the nginx config file ip addresses to match your setup*

### Check the health of the config file settings:
```
sudo nginx -t
```
### Reload nginx if all is ok:
```
sudo /etc/nginx/init.d/nginx reload
```

### Automating nginx

Check the status of nginx

```
sudo systemctl status nginx
```

Start the server
```
sudo systemctl start nginx
```

Configure nginx to automatically start on system startup
```
sudo systemctl enable nginx
```

### Run Starlette app on Uvicorn server under supervisor

Included in the directory is a `startServer.sh` bash script that just calls `imgapp.py` in the correct manner. 
Supervisor should be set up to run this script on startup and in the case of failures. Install and set it up with the following:
```
sudo apt-get install supervisor
cp EAPImgRotnWebApp.conf /etc/supervisor/conf.d / # replace the environment variable with your python path
sudo service supervisor stop
sudo service supervisor start
```

Check supervisor is running the server correctly with 
```
sudo supervisorctl
```

 
