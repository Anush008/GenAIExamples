# Language Translation

Language Translation is the communication of the meaning of a source-language text by means of an equivalent target-language text.

The workflow falls into the following architecture:

![architecture](https://i.imgur.com/VmoVugU.png)

# Start Backend Service

1. Start the TGI service to deploy your LLM

```sh
cd serving/tgi_gaudi
bash build_docker.sh
bash launch_tgi_service.sh
```

`launch_tgi_service.sh` by default uses `8080` as the TGI service's port. Please replace it if there are any port conflicts.

2. Start the Language Translation service

```sh
cd langchain/docker
bash build_docker.sh
docker run -it --name translation_server --net=host --ipc=host -e TGI_ENDPOINT=${TGI ENDPOINT} -e HUGGINGFACEHUB_API_TOKEN=${HUGGINGFACE_API_TOKEN} -e SERVER_PORT=8000 -e http_proxy=${http_proxy} -e https_proxy=${https_proxy} translation:latest bash
```

Here is the explanation of some of the above parameters:

- `TGI_ENDPOINT`: The endpoint of your TGI service, usually equal to `<ip of your machine>:<port of your TGI service>`.
- `HUGGINGFACEHUB_API_TOKEN`: Your HuggingFace hub API token, usually generated [here](https://huggingface.co/settings/tokens).
- `SERVER_PORT`: The port of the Translation service on the host.

3. Quick test

```sh
curl http://localhost:8000/v1/translation \
    -X POST \
    -d '{"language_from": "English","language_to": "Chinese","source_language": "\n我爱机器翻译。\n"}' \
    -H 'Content-Type: application/json'
```