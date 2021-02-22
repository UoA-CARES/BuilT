
import os
import torch
import numpy as np
import logging
import imgkit
import matplotlib.pyplot as plt

from built.forward_hook import ForwardHookBase
from built.registry import Registry
import tokenizers
@Registry.register(category='hooks')
class CAMForwardHook(ForwardHookBase):
    def forward(self, inputs, model, is_train):
        pr = {0: 'NEUTRAL', 1: 'POSITIVE', 2: 'NEGATIVE'}

        params = list(model.parameters())
        weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

        ids = inputs['ids']
        masks = inputs['masks']
        
        sentiment = inputs['sentiment_id']
        tweet = inputs['tweet']
        targets = inputs['sentiment_target']
        selected_text = inputs['selected_text']
        char_cent = inputs['char_centers']

        hidden_states, outputs = model(
            input_ids=ids,
            attention_mask=masks,
        )
        
        final_outputs = []
        masks = masks.cpu().detach().numpy().tolist()
        char_cent = char_cent.cpu().detach().numpy().tolist()

        tokenizer = tokenizers.ByteLevelBPETokenizer(
            vocab_file='tweet/input/roberta-base/vocab.json',
            merges_file='tweet/input/roberta-base/merges.txt',
            lowercase=True,
            add_prefix_space=True)
            
        for index in range(0, ids.shape[0]):
            last_conv_output = hidden_states[index]            
            last_conv_output = np.squeeze(last_conv_output)
            last_conv_output = torch.sigmoid(
                last_conv_output).cpu().detach().numpy().tolist()
            
            pred_vec = outputs[index]
            pred_vec = torch.sigmoid(pred_vec).cpu().detach().numpy().tolist()
            pred = np.argmax(pred_vec)
            layer_weights = weight_softmax[pred, :]
            
            final_output = np.dot(last_conv_output, layer_weights)
            
            # plt.figure(figsize=(20, 3))

            idx = np.sum(masks[index])
            v = np.argsort(final_output[:idx-1])

            mx = final_output[v[-1]]
            x = max(-10, -len(v))
            mn = final_output[v[x]]

            # plt.plot(char_cent[index][:idx-2], final_output[1:idx-1], 'o-')
            # plt.plot([1, 95], [mn, mn], ':')
            # plt.xlim((0, 95))
            # plt.yticks([])
            # plt.xticks([])
            # plt.title(
            #     f'Predict label is {pr[pred]} True label is {sentiment[index]} : {tweet[index].lower()}', size=16)
            # plt.savefig(f'{index}.png')
            # plt.close()

            # DISPLAY ACTIVATION TEXT
            html = ''
            for j in range(1, idx):
                x = (final_output[j]-mn)/(mx-mn)
                html += "<span style='background:{};font-family:monospace'>".format(
                    'rgba(75,204,247,%f)' % x)
                html += tokenizer.decode([ids[index][j]])
                html += "</span>"
            html += f" ({tokenizer.decode(sentiment[index].cpu().detach().numpy())})"
            # display(HTML(html))
            # html += ""
            # # DISPLAY TRUE SELECTED TEXT
            # cur_tweet = " ".join(tweet[index].lower().split())
            # cur_selected_text = " ".join(selected_text[index].lower().split())
            # sp = cur_tweet.split(cur_selected_text)
            # html += "<span style='font-family:monospace'>"+sp[0]+"</span>"
            # for j in range(1, len(sp)):
            #     html += "<span style='background:yellow;font-family:monospace'>"+cur_selected_text+'</span>'
            #     html += "<span style='font-family:monospace'>"+sp[j]+"</span>"
            # html += " (true)"

            options = {"xvfb": ""}
            try:
                pdir = 'cam_sentiemnt'
                if not os.path.exists(pdir):
                    os.makedirs(pdir)
                imgkit.from_string(html, f'{pdir}/{index}t.png', options=options)
            except:
                print(f"index={index}")

        return outputs
