#!/usr/bin/env python3
import json
import os
import numpy as np
import tensorflow as tf
import model, sample, encoder
import signal
import sys
import random
from flask import Flask, request, jsonify
from waitress import serve
from flask_cors import CORS
import time
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Reduce tensorflow logging
tf.get_logger().setLevel('ERROR') # Reduce tensorflow logging

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

def signal_handler(sig, frame):
    print('Quitting..')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

# Select the model to use, should match the file downloaded in the Dockerfile
model_to_use='1558M'
# model_to_use='774M'
# model_to_use='345M'
# model_to_use='124M'

generated_length=140 # length of the output,  If the length is None, then the number of tokens is decided by model hyperparameters
# generated_length=None # represents the number of tokens in the generated text. If the length is None, then the number of tokens is decided by model hyperparameters
seed_to_use=random.randrange(9_999_999)
samples_to_generate=1
batch_size_to_use=1
temp_to_use=0.75 #Default=1 temperature: This controls randomness in Boltzmann distribution. Lower temperature results in less random completions. As the temperature approaches zero, the model will become deterministic and repetitive. Higher temperature results in more random completions
top_k_value=0 #Default=0 top_k: This parameter controls diversity. If the value of top_k is set to 1, this means that only 1 word is considered for each step (token). If top_k is set to 40, that means 40 words are considered at each step. 0 (default) is a special setting meaning no restrictions. top_k = 40 generally is a good value
# print("%d. %s appears %d times." % (i, key, wordBank[key]))

class GPT2: 
    def __init__(self): 
        self.context = None
        self.sess = None
        self.enc = None
        self.output = None
        self.interact_model(
            model_to_use,
            seed_to_use,
            samples_to_generate,
            batch_size_to_use,
            generated_length,
            temp_to_use,
            top_k_value,
            '../models' # path to parent folder containing model subfolders
        )
    
    def interact_model(
        self,
        model_name,
        seed,
        nsamples,
        batch_size,
        length,
        temperature,
        top_k,
        models_dir
    ):
        models_dir = os.path.expanduser(os.path.expandvars(models_dir))
        if batch_size is None:
            batch_size = 1
        assert nsamples % batch_size == 0

        enc = encoder.get_encoder(model_name, models_dir)
        self.enc = enc
        hparams = model.default_hparams()
        with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))

        if length is None:
            length = hparams.n_ctx // 2
        elif length > hparams.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx) #TODO: change to print?

        print('⏳ Starting tensorflow session..')

        self.sess = tf.Session()
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        print('⏳ Loading %s model...' % (model_to_use))
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(self.sess, ckpt)

        self.context = context
        self.output = output
        return

def send_prompt(wp, sess, context, enc, output):
    print('🦾 Working on: "%s"..' % (wp))
    context_tokens = enc.encode(wp)
    out = sess.run(output, feed_dict={
            context: [context_tokens for _ in range(1)]
        })[:, len(context_tokens):]
    text = enc.decode(out[0])
    text = text.rsplit('.', 1)[0] + "." # Delete everything after last period, else you end up with half-sentences due to the `generated_length` cut-off
    text = text.replace('<|endoftext|>', ' ') # Replace this artifact which I dont think should be in the data, but occasionally appears
    return(text)
    
def load_tf(): 
    return GPT2()

@app.route("/wp", methods=['POST'])
def writing_prompt():
    input_json = request.get_json(force=True)
    # check for valid input
    if request.method == 'POST':
        if input_json.get("wp"):
            prompt = input_json.get("wp")
            # print(prompt)
            return(jsonify({'prompt': prompt, 'result': send_prompt(prompt, t.sess, t.context, t.enc, t.output), 'utc': datetime.utcnow()}), 200)
        else:
            return(jsonify({'error': 'Please specify a wp value'}), 404)
    else:
        return(jsonify({'error': 'Please POST to this endpoint'}), 404)

def main():
    print("👾 GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("🐶 Started server!")
    global t
    t = load_tf() # Init the class
    serve(app, port=2666, threads=8)

if __name__ == "__main__":
    main()
