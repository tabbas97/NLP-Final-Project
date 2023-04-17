import sys
import pandas as pd
import logging
import json

# import Rouge
from rouge import Rouge

from transformers import pipeline

def makePegasus():

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    pegasus = {
        "tokenizer" : AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail"),
        "model" : AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-cnn_dailymail")
    }

    pegasus['input_transform'] = lambda x : pegasus['tokenizer'](
        x,
        max_length=512,
        return_tensors="pt"
    )

    pegasus['summarize'] = lambda x : pegasus['model'].generate(
        pegasus['input_transform'](x)['input_ids'],
        num_beams=4,
        max_length=512,
        early_stopping=True
        )

    pegasus['decode'] = lambda x : pegasus['tokenizer'].batch_decode(
        pegasus['summarize'](x),
        skip_special_tokens=True
    )

    return pegasus

def makeProphetNet():

    from transformers import ProphetNetTokenizer, ProphetNetForConditionalGeneration, ProphetNetConfig

    prophetnet = {
        'tokenizer' : ProphetNetTokenizer.from_pretrained("microsoft/prophetnet-large-uncased-cnndm"),
        'model' : ProphetNetForConditionalGeneration.from_pretrained("microsoft/prophetnet-large-uncased-cnndm"),
    }

    prophetnet['input_transform'] = lambda x : prophetnet['tokenizer'](
        x,
        max_length=100,
        return_tensors="pt"
    )

    prophetnet['summarize'] = lambda x : prophetnet['model'].generate(
        prophetnet['input_transform'](x)['input_ids'],
        num_beams=4, 
        max_length=512, 
        early_stopping=True
        )

    prophetnet['decode'] = lambda x : prophetnet['tokenizer'].batch_decode(
        prophetnet['summarize'](x),
        skip_special_tokens=True
    )

    return prophetnet

def makeGPT2():

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("gavin124/gpt2-finetuned-cnn-summarization-v2")
    model = AutoModelForCausalLM.from_pretrained("gavin124/gpt2-finetuned-cnn-summarization-v2")

    gpt2 = {
        'tokenizer' : tokenizer,
        'model' : model,
    }

    gpt2['input_transform'] = lambda x : gpt2['tokenizer'](
        x,
        max_length=512,
        return_tensors="pt"
    )

    gpt2['summarize'] = lambda x : gpt2['model'].generate(
        gpt2['input_transform'](x)['input_ids'],
        num_beams=4,
        max_length=513,
        early_stopping=True
        )

    gpt2['decode'] = lambda x : gpt2['tokenizer'].batch_decode(
        gpt2['summarize'](x),
        skip_special_tokens=True
    )

    print(type(gpt2['input_transform']))
    print(type(gpt2['summarize']))
    print(type(gpt2['decode']))

    return gpt2

def makeBART():
    bart = {
        'model' :  pipeline('summarization', model='facebook/bart-large-cnn', max_length = 1024)
    }
    bart['decode'] = lambda x : bart['model'](x, truncation=True)

    return bart

def makeT5Small():

    t5 = {}

    t5['model'] = pipeline('summarization', model='t5-small')
    t5['decode'] = lambda x : t5['model'](x)

    return t5

def makeT5Large():
    
    t5 = {}
    
    t5['model'] = pipeline('summarization', model='t5-large')
    t5['decode'] = lambda x : t5['model'](x)
    
    return t5

def makeBrio():
    logging.warning(
        "This model has a high memory footprint. Will most likely kill the kernel."
        "32 GB of RAM is recommended."
        )

    from transformers import BartTokenizer, BartForConditionalGeneration
 
    tokenizer = BartTokenizer.from_pretrained("Yale-LILY/brio-cnndm-uncased")
    model = BartForConditionalGeneration.from_pretrained("Yale-LILY/brio-cnndm-uncased")

    brio = {
        'tokenizer' : tokenizer,
        'model' : model,
    }

    brio['input_transform'] = lambda x : brio['tokenizer'](
        [val.lower() for val in x],
        max_length=1024,
        return_tensors="pt"
    )

    brio['summarize'] = lambda x : brio['model'].generate(
        brio['input_transform'](x)['input_ids'],
        num_beams=4,
        max_length=513,
        early_stopping=True
        )

    brio['decode'] = lambda x : brio['tokenizer'].batch_decode(
        brio['summarize'](x),
        skip_special_tokens=True
    )

    return brio

modelChoices = {
    'pegasus' : makePegasus,
    'prophetnet' : makeProphetNet,
    'gpt2' : makeGPT2,
    'bart' : makeBART,
    't5-small' : makeT5Small,
    't5-large' : makeT5Large,
    'brio' : makeBrio
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='pegasus', choices=modelChoices.keys())
    parser.add_argument('--input', type=str, default='val_sample.csv')
    parser.add_argument('--output', type=str, default='val_sample_summaries.csv')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    inputDF = pd.read_csv(args.input)

    inputSeq = inputDF['article'].tolist()
    outputs = inputDF['highlights'].tolist()

    summModel = modelChoices[args.model]()

    predictions = []

    for i in range(len(inputSeq)):
        summary = summModel['decode'](inputSeq[i])
        if args.verbose:
            print(json.dumps({
                'article' : inputSeq[i],
                'highlights' : outputs[i],
                'prediction' : summary
            }, indent=4))

        predictions.append(summary)
        rouge = Rouge()
        scores = rouge.get_scores(summary, outputs[i])

    outputDF = pd.DataFrame({'article' : inputSeq, 'highlights' : outputs, 'predictions' : predictions})
    outputDF.to_csv(args.output, index=False)