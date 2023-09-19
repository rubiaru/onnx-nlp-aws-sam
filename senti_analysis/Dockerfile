FROM public.ecr.aws/lambda/python:3.9

COPY requirements.txt ./
RUN python3.9 -m pip install -r requirements.txt -t .

COPY app.py ./
COPY my_model_convert.onnx ./
COPY bert-base-multilingual-cased/tokenizer.json ./bert-base-multilingual-cased/tokenizer.json

# Command can be overwritten by providing a different command in the template directly.
CMD ["app.lambda_handler"]
