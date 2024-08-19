# NOTES


## scripts
Need these exports so `pip` process will find the libraries for `OpenFST`
```shell
export LDFLAGS=-L/usr/local/lib
export CPPFLAGS=-I/usr/local/include/
```


```shell
pip install nemo_toolkit[all] torch transformers pytorch_lightning
```


```shell
pip install nemo_toolkit[all] torch transformers pytorch_lightning

```


```shell
# BUILD
docker build -t main .

# docker run -it main sh

docker run -it --rm --name my-running-app main

# deploy the image to GCP
gcloud config set project centifichomework
gcloud run deploy centific-homework --port 8080 --source .
```