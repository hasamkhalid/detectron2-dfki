FROM python:3.8

ADD file:4974bb5483c392fb54a35f3799802d623d14632747493dce5feb4d435634b4ac in / 
ADD requirements.txt /
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
RUN pip install -r /requirements.txt
ADD main_eval.py /
ADD LossEvalHook.py /
ADD detectron2 /
ADD dataverse_files /
ADD output /
ADD out_images /
ENV PYTHONUNBUFFERED=1
CMD [ "python", "./main_eval.py" ]