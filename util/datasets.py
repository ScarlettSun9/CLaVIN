# coding=utf-8
# Copyright 2022 Gen Luo. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import  json, re,random
import pandas as pd
import torch.utils.data as Data
from torchvision.transforms import transforms
import os
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from util.base_prompt import *
import torch
from lavin import Tokenizer
import copy

class ScienceQADataSet(Data.Dataset):
    def __init__(self, args, split, model_path, max_words=512, max_image_feats=1):
        super(ScienceQADataSet, self).__init__()
        self.args = args
        # --------------------------
        # ---- Raw data loading ---
        # --------------------------
        self.problems = json.load(open(os.path.join(args.data_root, 'problems.json')))
        pid_splits = json.load(open(os.path.join(args.data_root, 'pid_splits.json')))
        captions = json.load(open(args.caption_file))["captions"]
        self.image_path=os.path.join(args.data_root,'images',split)
        self.tokenizer = Tokenizer(model_path=model_path + '/tokenizer.model') # Here tokenizer path is llama/tokenizer.model, then the path is passed to Tokenizer.
        self.max_words = max_words
        self.max_image_feats=max_image_feats
        self.split=split
        for qid in self.problems:
            self.problems[qid]['caption'] = captions[qid] if qid in captions else ""

        self.qids = pid_splits['%s' % (split)] # Here it is image ['train'] ids in pid_split.json because the split here is 'train' from main.py

        print(f"number of problems in split {split}: {len(self.qids)}\n")

        self.transforms=transforms.Compose([transforms.Resize((224, 224), interpolation=Image.BICUBIC),transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

    def tokenize(self,prompt,answer):
        example=prompt+answer
        # print(prompt)
        prompt=torch.tensor(self.tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.int64) # prompt [1 2 3]
        example = torch.tensor(self.tokenizer.encode(example, bos=True, eos=True), dtype=torch.int64) # example [1 2 3 4 5 6]
        padding = self.max_words - example.shape[0] # assume padding > 0 and example [1 2 3 4 5 6 -1 -1 -1]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1)) # Concatenate the example and padding, which is -1 here isntead of 0.
        elif padding < 0:
            example = example[:self.max_words]
        labels = copy.deepcopy(example) # labels [1 2 3 4 5 6 -1 -1 -1]
        labels[:len(prompt)] = -1     # set all the prompt label as -1 --> labels [-1 -1 -1 4 5 6 -1 -1 -1]
        example_mask = example.ge(0)    
        # example_mask is a tensor contains True or False (for all elements in example, whether it is greater or equal to 0)
        # example_mask [F F F T T T F F F]
        label_mask = labels.ge(0) # label_mask [T T T T T T F F F]
        example[~example_mask] = 0      
        # set all False in example_mask position as 0. I think the whole process is the same as set the padding = 0
        # example [0 0 0 4 5 6 0 0 0]
        labels[~label_mask] = 0 # labels [-1 -1 -1 4 5 6 0 0 0]
        example_mask = example_mask.float() # example_mask: [0.0  0.0  0.0  1.0  1.0  1.0  0.0  0.0  0.0]
        label_mask = label_mask.float() # label_mask: [1.0  1.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0]
        return example, labels, example_mask,label_mask


    def __getitem__(self, idx):

        prompt_question, prompt_answer = build_prompt(self.problems, self.qids[idx], self.args)
        answer, choices, qid=self.problems[self.qids[idx]]["answer"], self.problems[self.qids[idx]]["choices"],self.qids[idx]

        if self.problems[self.qids[idx]]['image'] is not None:
            image = Image.open(os.path.join(self.image_path, self.qids[idx], 'image.png')).convert('RGB')
            image = self.transforms(image)
            image_mask=torch.cat([torch.Tensor([float('-inf')]*self.max_image_feats),torch.zeros(self.max_words)])
            indicator=1
        else:
            image=torch.Tensor(torch.zeros(3,224,224).float())
            image_mask=torch.zeros(self.max_words+self.max_image_feats)
            indicator=0

        example, labels, example_mask, label_mask=self.tokenize(prompt_question,prompt_answer)

        return example, labels, example_mask, image, indicator

    def __len__(self):
        return len(self.qids)

    def shuffle_list(self, list):
        random.shuffle(list)



class InstrcutDataSet(Data.Dataset):
    def __init__(self, args,split,model_path,max_words=512,max_image_feats=1):
        super(InstrcutDataSet, self).__init__()
        self.args = args
        # --------------------------
        # ---- Raw data loading ---
        # --------------------------
        self.data = json.load(open(os.path.join(args.data_root, 'all_data.json')))[split]

        self.tokenizer = Tokenizer(model_path=model_path + '/tokenizer.model')
        self.max_words = max_words
        self.max_image_feats=max_image_feats
        self.split=split
        self.qids = [item['qid'] for item in self.data]

        print(f"number of problems in split {split}: {len(self.qids)}\n")

        self.transforms=transforms.Compose([transforms.Resize((224, 224), interpolation=Image.BICUBIC),transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

    def tokenize(self,prompt,answer,max_words=512):
        example=prompt+answer
        # print(prompt)
        prompt=torch.tensor(self.tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.int64)
        example = torch.tensor(self.tokenizer.encode(example, bos=True, eos=True), dtype=torch.int64)
        padding = max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[:self.max_words]
        labels = copy.deepcopy(example)
        labels[:len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()
        return example, labels, example_mask,label_mask


    def __getitem__(self, idx):

        prompt_question=self.data[idx]['instruction']
        prompt_answer=self.data[idx]['answer']

        if self.data[idx]['image'] is not None:
            # image_path='../data/images/train' if self.data[idx]['image_source']=='sqa' else '../data/images/train2014'
            if self.data[idx]['image_source'] == 'sqa':
                image = Image.open(os.path.join('../data/images/train', self.qids[idx], 'image.png')).convert('RGB')
            else:
                image = Image.open(os.path.join('../data/images/train2014',   'COCO_train2014_'+self.data[idx]['image'])).convert('RGB')
            image = self.transforms(image)
            indicator=1
        else:
            image=torch.Tensor(torch.zeros(3,224,224).float())
            indicator=0

        # print(prompt_question,prompt_answer)
        example, labels, example_mask, label_mask=self.tokenize(prompt_question,prompt_answer)

        return example, labels, example_mask, image,indicator

    def __len__(self):
        return len(self.qids)

    def shuffle_list(self, list):
        random.shuffle(list)


########################################################
########################################################
########################################################

class PretrainDataSet(Data.Dataset):
    def __init__(self, args, data_path, tokenizer_path, max_words=512, max_image_feats=1):
        super(PretrainDataSet, self).__init__()
        self.args = args
        # --------------------------
        # ---- Raw data loading ---
        # --------------------------
        self.data = pd.read_csv(os.path.join(args.data_path, 'PT_total_final_mini.csv'))['content'].tolist()
        self.imageIDs = pd.read_csv(os.path.join(args.data_path, 'PT_total_final_mini.csv'))['img'].tolist()

        self.tokenizer = Tokenizer(tokenizer_path + '/tokenizer.model')
        self.max_words = max_words
        self.max_image_feats=max_image_feats
        self.qids = [str(i) for i in range(len(self.data))]


        self.transforms=transforms.Compose([transforms.Resize((224, 224), interpolation=Image.BICUBIC),transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

    def tokenize(self, sentence, max_words=512):
        example = torch.tensor(self.tokenizer.encode(sentence, bos=True, eos=True), dtype=torch.int64)
        padding = max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[:self.max_words]
        labels = copy.deepcopy(example)
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()
        return example, labels, example_mask,label_mask


    def __getitem__(self, idx):
 
        text = str(self.data[idx])
        if text == 'nan':
            text = '请讨论和舆情有关的内容，谢谢！'
        image_id = self.imageIDs[idx]
        image_name = str(image_id)
        image_folder = os.path.join(self.args.data_path, 'PT_Dataset')
        assumed_image_path = os.path.join(image_folder, image_name)
        if os.path.exists(assumed_image_path):
            try:
                image = Image.open(assumed_image_path).convert('RGB')
                image = self.transforms(image)
                indicator = 1
            except:
                image = torch.Tensor(torch.zeros(3,224,224).float())
                indicator = 0
        else:
            image = torch.Tensor(torch.zeros(3,224,224).float())
            indicator = 0

        # print(prompt_question,prompt_answer)
        example, labels, example_mask, label_mask=self.tokenize(text)

        return example, labels, example_mask, image, indicator

    def __len__(self):
        return len(self.qids)

    def shuffle_list(self, list):
        random.shuffle(list)

########################################################
########################################################
########################################################












########################################################
########################################################
########################################################

class OA_DataSet(Data.Dataset):
    def __init__(self, args, model_path, max_words=512, max_image_feats=1):
        super(ScienceQADataSet, self).__init__()
        self.args = args
        # --------------------------
        # ---- Raw data loading ---
        # --------------------------
        captions = json.load(open(args.caption_file))["captions"]
        pid_splits = json.load(open(os.path.join(args.data_root, 'pid_splits.json')))
        self.problems = json.load(open(os.path.join(args.data_root, 'problems.json')))
        self.image_path=os.path.join(args.data_root,'SFT_Dataset')
        self.tokenizer = Tokenizer(tokenizer_path + '/tokenizer.model')
        self.max_words = max_words
        self.max_image_feats=max_image_feats
        self.split=split
        for qid in self.problems:
            self.problems[qid]['caption'] = captions[qid] if qid in captions else "" # Add caption to the problem.json

        self.qids = pid_splits['%s' % (split)] # Here it is image ['train'] ids in pid_split.json because the split here is 'train' from main.py

        print(f"number of problems in split {split}: {len(self.qids)}\n")

        self.transforms=transforms.Compose([transforms.Resize((224, 224), interpolation=Image.BICUBIC),transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

    def tokenize(self,prompt,answer):
        example=prompt+answer
        # print(prompt)
        prompt=torch.tensor(self.tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.int64) # prompt [1 2 3]
        example = torch.tensor(self.tokenizer.encode(example, bos=True, eos=True), dtype=torch.int64) # example [1 2 3 4 5 6]
        padding = self.max_words - example.shape[0] # assume padding > 0 and example [1 2 3 4 5 6 -1 -1 -1]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1)) # Concatenate the example and padding, which is -1 here isntead of 0.
        elif padding < 0:
            example = example[:self.max_words]
        labels = copy.deepcopy(example) # labels [1 2 3 4 5 6 -1 -1 -1]
        labels[:len(prompt)] = -1     # set all the prompt label as -1 --> labels [-1 -1 -1 4 5 6 -1 -1 -1]
        example_mask = example.ge(0)    
        # example_mask is a tensor contains True or False (for all elements in example, whether it is greater or equal to 0)
        # example_mask [F F F T T T F F F]
        label_mask = labels.ge(0) # label_mask [T T T T T T F F F]
        example[~example_mask] = 0      
        # set all False in example_mask position as 0. I think the whole process is the same as set the padding = 0
        # example [0 0 0 4 5 6 0 0 0]
        labels[~label_mask] = 0 # labels [-1 -1 -1 4 5 6 0 0 0]
        example_mask = example_mask.float() # example_mask: [0.0  0.0  0.0  1.0  1.0  1.0  0.0  0.0  0.0]
        label_mask = label_mask.float() # label_mask: [1.0  1.0  1.0  1.0  1.0  1.0  0.0  0.0  0.0]
        return example, labels, example_mask,label_mask


    def __getitem__(self, idx):

        prompt_question, prompt_answer = build_prompt(self.problems, self.qids[idx], self.args)
        answer, choices, qid=self.problems[self.qids[idx]]["answer"], self.problems[self.qids[idx]]["choices"],self.qids[idx]

        if self.problems[self.qids[idx]]['image'] is not None:
            image = Image.open(os.path.join(self.image_path, self.qids[idx], 'image.png')).convert('RGB') # Change here!!!
            image = self.transforms(image)
            image_mask=torch.cat([torch.Tensor([float('-inf')]*self.max_image_feats),torch.zeros(self.max_words)])
            indicator=1
        else:
            image=torch.Tensor(torch.zeros(3,224,224).float())
            image_mask=torch.zeros(self.max_words+self.max_image_feats)
            indicator=0

        example, labels, example_mask, label_mask=self.tokenize(prompt_question,prompt_answer)

        return example, labels, example_mask, image, indicator

    def __len__(self):
        return len(self.qids)

    def shuffle_list(self, list):
        random.shuffle(list)

########################################################
########################################################
########################################################









if __name__ == '__main__':
    from torch.utils.data import DataLoader
    class Cfg():
        def __init__(self):
            super(Cfg, self).__init__()
            self.options = ["A", "B", "C", "D", "E"]
            self.use_caption = True
            self.prompt_format = 'CQM-A'
            self.data_root = './data'
            self.output_root = './output'
            self.caption_file = './data/captions.json'
    cfg=Cfg()
    dataset=ScienceQADataSet(cfg,'val','./data/weights')
    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=True)
    max_question_len=0
    max_answer_len=0
    #406 max question
    for prompt_questions,question_mask,images,image_masks,prompt_answers,answers,qids in data_loader:
        print(prompt_questions)
        print(answers)
    #     if len(prompt_questions[0].split())>max_question_len:
    #         max_question_len=len(prompt_questions[0].split())
    #     if len(prompt_answers[0].split())>max_answer_len:
    #         max_answer_len=len(prompt_answers[0].split())
    # print(max_question_len,max_answer_len)






