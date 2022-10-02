FROM python:3.7.1
RUN pip install pyyaml==5.3.1
RUN pip install -U torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/cu101/torch_stable.html
RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html
ADD requirements.txt /
RUN pip install -r /requirements.txt
ADD main_eval.py /
ADD LossEvalHook.py /
ADD detectron2 /
ADD dataverse_files /
ADD output /
ADD out_images /
ENV PYTHONUNBUFFERED=1
CMD [ "python", "./main_eval.py" ]
