FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

RUN rm -rf /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx build-essential wget git curl libsm6 libxrender1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install anaconda
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh --output conda_installer.sh
RUN /bin/bash conda_installer.sh -b -p /opt/conda \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete \
    && /opt/conda/bin/conda clean -afy
RUN rm conda_installer.sh

ENV PATH=/opt/conda/bin:$PATH
ENV CUDA_HOME=/usr/local/cuda

# conda install critical libraries
COPY .condarc /root/.condarc
RUN conda install -y numpy=1.26.4 scikit-learn=1.5.1 scipy=1.14.1 pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge

# pip install other libraries
RUN pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt /workspace/requirements.txt
RUN pip install datasets==2.16.1 \
                datasketch==1.6.4 \
                Flask==3.0.3 \
                Flask_Cors==4.0.0 \
                jieba==0.42.1 \
                jsonlines==4.0.0 \
                marshmallow==3.22.0 \
                matplotlib==3.5.1 \
                ngrok==1.4.0 \
                nltk==3.8 \
                openai==1.42.0 \
                pandas==1.5.3 \
                peft==0.7.1 \
                psutil==5.9.8 \
                pydantic==2.8.2 \
                rich==13.7.1 \
                sentence_transformers==2.3.1 \
                simhash==2.1.2 \
                tiktoken==0.5.1 \
                transformers==4.44.0 \
                jinja2==3.1.2 \
                jsonlines==4.0.0 \
                trl==0.11.3 \
                ujson==5.1.0 \
                wandb==0.18.3 --no-cache-dir
RUN pip install --no-cache-dir copious==0.1.24 easydict==1.13 pytest==8.3.3 pytest-cov loguru tqdm

# If not use docker build, we need to temporarily put cuda (of correct version) to /usr/loca/cuda or conda install cuda-toolkit cuda-cudart cuda-cccl libcublas libcusparse libcusolver
RUN pip install --no-cache-dir flash-attn==0.2.2

# install libgllib2.0 and set TimeZone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
RUN apt-get update \
   && apt-get install -y libglib2.0-0 libxext6 tzdata  \
   && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
   && echo $TZ > /etc/timezone \
   && dpkg-reconfigure -f noninteractive tzdata \
   && apt-get clean \
   && rm -rf /var/lib/apt/lists/*

# install pytorch3d
# RUN pip install --no-cache-dir --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.4.1cu121
# COPY ./pytorch3d-0.7.8+pt2.4.1cu121-cp310-cp310-linux_x86_64.whl /pytorch3d-0.7.8+pt2.4.1cu121-cp310-cp310-linux_x86_64.whl
# RUN pip install --no-cache-dir /pytorch3d-0.7.8+pt2.4.1cu121-cp310-cp310-linux_x86_64.whl
# RUN rm -rf /pytorch3d-0.7.8+pt2.4.1cu121-cp310-cp310-linux_x86_64.whl

# COPY libstdc++.so.6 to system folder to prevent complaining GLIBCXX_3.4.29 not found
RUN cp /opt/conda/lib/libstdc++.so.6 /lib/x86_64-linux-gnu/

# install extra python packages

WORKDIR /workspace

CMD [ "/bin/bash"  ]

