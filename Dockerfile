FROM python:3.7.1
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip
RUN pip install opencv-python
RUN pip install pyyaml==5.3.1
RUN pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html
ADD requirements.txt /
RUN pip install -r /requirements.txt
RUN pip install fairscale
RUN pip install timm
ADD LossEvalHook.py /
ADD dataverse_files /
ADD output /
ADD out_images /
ADD main_eval.py /
ADD detectron2 /
ENV PYTHONUNBUFFERED=1
COPY detectron2/ detectron2/
COPY dataverse_files/ dataverse_files/
COPY out_images/ out_images/
COPY output/ output/

CMD [ "python", "./main_eval.py" ]
