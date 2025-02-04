FROM python:3.10-slim
WORKDIR /codeBase
COPY /images/codeBase_CarPricePrediction/RunAIModel.py /codeBase/RunAIModel.py
COPY /images/codeBase_CarPricePrediction/RunOLSModel.py /codeBase/RunOLSModel.py
RUN pip install --no-cache-dir numpy pandas tensorflow==2.18.0 keras statsmodels
# CMD ["python", "/tmp/codeBase/RunAIModel.py"]