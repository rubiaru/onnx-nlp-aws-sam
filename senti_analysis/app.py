
from tokenizers import Tokenizer
import onnxruntime as ort
import numpy as np
import json
import numpy as np

from io import BytesIO

# sequence max length
MAX_LEN = 128

tokenizer = Tokenizer.from_file("/var/task/bert-base-multilingual-cased/tokenizer.json")

def lambda_handler(event, context):
      
    body = json.loads(event['body'])

    sentences = body["sentences"].split("\n")

    opt = ort.SessionOptions()
    opt.graph_optimization_level= ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    opt.log_severity_level=3
    opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    # make a model using bert_naver_movie.ipynb
    ort_session = ort.InferenceSession('my_model_convert.onnx', opt)

    encoded_input = tokenizer.encode(sentences[0])

    encoded_input = { 
                      "input_ids" : np.atleast_2d(encoded_input.ids),
                      "token_type_ids" : np.atleast_2d(encoded_input.type_ids),
                      "attention_mask" : np.atleast_2d(encoded_input.attention_mask)
                    }
    
    result = ort_session.run(None, encoded_input)[0]

    dicResult = {
            'statusCode': 200,
            'body': json.dumps(
                {
                    "sentences" : sentences,
                    "neg" : str(result[0][0]),
                    "pos" : str(result[0][1]),
                    "result" : str(np.argmax(result))
                }
            )
        }

    return dicResult
