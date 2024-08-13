INPUT_SCHEMA = {
    "prompt": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["What is deep learning?"]
    },
    "system_prompt": {
        'datatype': 'STRING',
        'required': False,
        'shape': [1],
        'example': ["You are a friendly bot."]
    },
    "temperature": {
        'datatype': 'FP32',
        'required': False,
        'shape': [1],
        'example': [0.7]
    },
    "top_p": {
        'datatype': 'FP32',
        'required': False,
        'shape': [1],
        'example': [0.1]
    },
    "repeat_penalty": {
        'datatype': 'FP32',
        'required': False,
        'shape': [1],
        'example': [1.18]
    },
    "max_tokens": {
        'datatype': 'INT16',
        'required': False,
        'shape': [1],
        'example': [512]
    },
    "top_k":{
        'datatype': 'INT8',
        'required': False,
        'shape': [1],
        'example': [40]
    }
}
