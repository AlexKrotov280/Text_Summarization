# Azure libraries
from logging.handlers import SMTPHandler
from azure.storage.blob import BlobServiceClient
from azure.servicebus import ServiceBusClient
# General libraries
import os
import pathlib
import sys
import datetime
import confuse
import logging
import re
import json
import requests

# ML libraries
from transformers import BertTokenizer, BertConfig
import torch
# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertForTokenClassification, AdamW
import numpy as np
# import spacy
from spacy.lang.en import English;

nlp = English()
nlp.add_pipe('sentencizer')
from scipy.special import softmax
from transformers import T5Tokenizer, T5ForConditionalGeneration
import nltk

nltk.download("all")
from nltk.tokenize import sent_tokenize

# Get the the config from YAML file
root_dir = os.getcwd()
config_file_path = os.path.join(root_dir, "config_default.yaml")

config = confuse.Configuration('textsummarization', __name__)
config.set_file(config_file_path)

# Get Azure Service Bus settings from config.yaml
#connection_string = config['azure_service_bus']['connections_str'].get()
topic_name = config['azure_service_bus']['topic_name'].get()
subscription_name = config['azure_service_bus']['subscription_name'].get()

# Get Azure Service Bus settings from environmental variables
connection_string = os.environ['CONNECTIONSTR']
url_first_part = os.environ['URLFIRST']
url_second_part = os.environ['URLSECOND']

# Get SMTP settings from config
smtpserver = config['SMTPhandler']['smtp_server'].get()
from_address = config['SMTPhandler']['from_address'].get()
to_address = config['SMTPhandler']['to_address'].get()
username = config['SMTPhandler']['username'].get()
password = config['SMTPhandler']['password'].get()

# Get API_URL settings from config.yaml
# url_first_part = config['HTTPrequest']['url_first_part']
# url_second_part = config['HTTPrequest']['url_second_part']


# Create folder name and file name for log file in Blob storage
current_time = str(datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%Z"))
hour = current_time.split("T")[1][:2]
folder_name = current_time.split("T")[0]
log_file_name = (hour + ".log").replace(":", "")
directory_name = "PythonLogs"
sub_directory_name = "TextSummarization"
blob_name = directory_name + "/" + sub_directory_name + "/" + folder_name + "/" + log_file_name

# Set up file and smtp loggers
logger_tofile = logging.getLogger(__name__)
logger_tofile.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler(log_file_name)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

logger_toemail = logging.getLogger("SMTP")
logger_toemail.setLevel(logging.INFO)
smtp_handler = SMTPHandler(mailhost=smtpserver,
                           fromaddr=from_address,
                           toaddrs=[to_address],
                           subject='Python NER testing',
                           credentials=(username, password), secure=None)
smtp_handler.setFormatter(formatter)

logger_tofile.addHandler(file_handler)
logger_toemail.addHandler(smtp_handler)

unique_tags = ['Collection', 'Season', 'Brand', 'Team', 'Color', 'O', 'Prod_t', 'SurfaceType', 'Gender', 'Strip', 'X',
               '[CLS]', '[SEP]']

tag2idx = {'Collection': 0, 'Season': 1, 'Brand': 2, 'Team': 3, 'Color': 4, 'O': 5, 'Prod_t': 6, 'SurfaceType': 7,
           'Gender': 8, 'Strip': 9, 'X': 10, '[CLS]': 11, '[SEP]': 12}

tag2name = {tag2idx[key]: key for key in tag2idx.keys()}

root_dir = os.getcwd()
save_model_address = os.path.join(root_dir, "model")

save_model = BertForTokenClassification.from_pretrained(save_model_address, num_labels=len(tag2idx))
tokenizer = BertTokenizer.from_pretrained(save_model_address, do_lower_case=False)

model = T5ForConditionalGeneration.from_pretrained('t5-base')
summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Get messages from a queue
def recieve_message_from_topic():
    messages = []
    with ServiceBusClient.from_connection_string(connection_string) as client:
        with client.get_subscription_receiver(topic_name, subscription_name, max_wait_time=10) as receiver:
            for msg in receiver:
                receiver.complete_message(msg)
                messages.append(str(msg))
    return messages


def logs_to_blob():
    azure_storage_connections_str = config['AzureStorage']['azure_storage_connections_str'].get()
    container_name = config['AzureStorage']['container_name'].get()

    # # Create the BlobServiceClient object which will be used to create a container client
    blob_service_client = BlobServiceClient.from_connection_string(azure_storage_connections_str)

    upload_file_path = log_file_name
    # Create a blob client using the local file name as the name for the blob
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    # Upload the created file
    with open(upload_file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    logging.shutdown()
    pathlib.Path(upload_file_path).unlink()


# Split summarized text into sentences and fix issues with upper cases
def split_sent_nltk(summarized_text):
    out_str = ""
    for sent in sent_tokenize(summarized_text):
        sent_list = list(sent)
        sent_list[0] = sent_list[0].upper()
        new_string = "".join(sent_list)
        out_str += " " + new_string
        out_str = out_str.strip(" ")
        if out_str[-1] != ".":
            out_str += "."
    return out_str


# BERT based text summarization
def summarization(text):
    if len(text) > 400:
        generator_len = 55
        ngram_size = 6
    elif len(text) > 300:
        generator_len = 35
        ngram_size = 6
    elif len(text) > 150:
        generator_len = 30
        ngram_size = 5
    else:
        generator_len = 25
        ngram_size = 4

    preprocess_text = text.strip().replace("\n", "")
    t5_prepared_text = "summarize: " + preprocess_text

    tokenized_text = summary_tokenizer.encode(t5_prepared_text, max_length=512, truncation=True,
                                              return_tensors="pt").to(device)

    summary_ids = model.to(device).generate(tokenized_text,
                                            num_beams=4,
                                            no_repeat_ngram_size=ngram_size,
                                            min_length=generator_len,
                                            max_length=500,
                                            early_stopping=True,
                                            length_penalty=3.0,
                                            num_return_sequences=3)
    output = summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return split_sent_nltk(output)


def text_to_sent(text):
    doc = nlp(text)
    sentences = [str(sent).strip() for sent in doc.sents]
    return sentences


def ner_inference(sentence):
    max_len = 45
    temp_token = []
    tokenized_texts = []
    temp_token.append('[CLS]')
    token_list = tokenizer.tokenize(sentence)
    for m, token in enumerate(token_list):
        temp_token.append(token)
    if len(temp_token) > max_len - 1:
        temp_token = temp_token[:max_len - 1]
    temp_token.append('[SEP]')
    token_vocab = {}
    tokenized_texts.append(temp_token)
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=max_len, dtype="long", truncating="post", padding="post")

    attention_masks = [[int(i > 0) for i in ii] for ii in input_ids]
    attention_masks[0];

    segment_ids = [[0] * len(input_id) for input_id in input_ids]
    segment_ids[0];

    input_ids = torch.tensor(input_ids).long()
    attention_masks = torch.tensor(attention_masks).long()
    segment_ids = torch.tensor(segment_ids).long()

    save_model.eval();

    with torch.no_grad():
        outputs = save_model(input_ids, token_type_ids=None,
                             attention_mask=None, )
        # For eval mode, the first result of outputs is logits
        logits = outputs[0]
    predict_results = logits.detach().cpu().numpy()

    result_arrays_soft = softmax(predict_results[0])

    result_array = result_arrays_soft

    result_list = np.argmax(result_array, axis=-1)
    result_list
    for i, mark in enumerate(attention_masks[0]):
        if mark > 0 and temp_token[i] != '[CLS]' and temp_token[i] != '[SEP]' and temp_token[i] != '.':
            token_vocab[temp_token[i]] = result_list[i]
    # print("!Token vocab! {}".format(token_vocab))
    return token_vocab


def ner_position(token_vocab, sent):
    tag_vocab = {}
    token_list = [0, 1, 2, 3, 4, 6, 7, 8, 9]

    for word_number, word in enumerate(sent.split(), start=1):
        for key, value in token_vocab.items():
            if value in token_list and key.replace("##", "") in word:
                if unique_tags[value] not in tag_vocab:
                    tag_vocab[unique_tags[value]] = word_number
    return tag_vocab


def ner_extraction(sentence, prod_name):
    sentence_lower = sentence.lower()
    prod_name_lower = prod_name.lower()
    position_list = []
    final_string = ""
    for word_number, word in enumerate(sentence_lower.split(), start=1):
        if re.search(r'\b' + word + r'\b', prod_name_lower):
            position_list.append(word_number)
    if position_list:
        first = min(position_list)
        last = max(position_list)
        for position in range(first - 1, last):
            final_string = final_string + " " + sentence.split()[position]
    return final_string.strip()


def phrase_substitution(final_string, target_string, sent):
    final_name = sent.replace(final_string, target_string)
    return final_name


def grammar_correction(sentence, target_string, product_type):
    product_list = ["Trainers", "Boots"]
    if product_type in product_list:
        # Split target string into words and extract brand and product names
        brand_name = target_string.split()[0]
        product_name = target_string.split()[-1]

        # Split sentence into words and count the indexes of brand and product
        word_in_sent = sentence.split()
        brand_name_index = word_in_sent.index(brand_name)
        product_name_index = word_in_sent.index(product_name)

        final_sentence = sentence

        if word_in_sent[product_name_index + 1] == "has":
            final_sentence = final_sentence.replace(" has", " have")
        elif word_in_sent[product_name_index + 1] == "is":
            final_sentence = final_sentence.replace(" is", " are")
        # else:
        #     target_word_original = " " + word_in_sent[product_name_index + 1]
        #     target_word_final = " " + word_in_sent[product_name_index + 1][:-1]
        #     final_sentence = final_sentence.replace(target_word_original, target_word_final)

        if word_in_sent[brand_name_index - 1] == "The":
            final_sentence = final_sentence.replace("The ", "These ")
    else:
        final_sentence = sentence
    return final_sentence


def ner(summarized_description, product_name, target_name, product_type):
    target_final_sentence = ""
    sent_list = text_to_sent(summarized_description)
    for sent in sent_list:
        token_vocab = ner_inference(sent)
        ner_vocab = ner_position(token_vocab, sent)
        final_string = ner_extraction(sent, product_name)
        if len(ner_vocab) > 1 and len(final_string.split(" ")) > 2:
            target_sent = phrase_substitution(final_string, target_name, sent)
            target_sent = grammar_correction(target_sent, target_name, product_type)
            target_final_sentence += " "
            target_final_sentence += target_sent
        else:
            target_final_sentence += " "
            target_final_sentence += sent
    return target_final_sentence.strip()


def create_target_name(attributes_list, product_type):
    # Trainers
    try:
        if product_type == "Trainers":
            brand = attributes_list["Brand"]
            collection = attributes_list["Collection"]
            gender = attributes_list["Gender"]
            target_name = brand + " " + collection + " " + gender + " Shoes "
        # Boots
        elif product_type == "Boots":
            brand = attributes_list["Brand"]
            #print("!", brand)
            collection = attributes_list["Collection"]
            sub_collection = attributes_list["SubCollection"]
            target_name = brand + " " + collection + " " + sub_collection + " Football Boots"
        # Replica shirt
        elif product_type == "Replica Shirt":
            brand = attributes_list["Brand"]
            team = attributes_list["Team"]
            gender = attributes_list["Gender"]
            target_name = brand + " " + team + " " + gender + " Shirt "
        # Jacker
        elif product_type == "Jacket":
            brand = attributes_list["Brand"]
            jacket_type = attributes_list["JacketType"]
            gender = attributes_list["Gender"]
            target_name = brand + " " + jacket_type + " " + gender + " Jacket "
        # Hoodie
        elif product_type == "Hoodie":
            brand = attributes_list["Brand"]
            gender = attributes_list["Gender"]
            target_name = brand + " " + gender + " Hoodie "
        # T-Shirt
        elif product_type == "T-Shirt":
            brand = attributes_list["Brand"]
            gender = attributes_list["Gender"]
            target_name = brand + " " + gender + " T-Shirt "
        else:
            target_name = None
    except Exception:
        target_name = ""
        logger_tofile.info('Target name creation attribute issue', exc_info=sys.exc_info())
    return target_name


def textprocessing():
    output_array = []
    product_list = ["Trainers", "Boots", "Replica Shirt", "Jacket", "Hoodie", "T-Shirt"]
    output_extra_json = {
        "path": "/AutoGeneratedDescriptionExpiry",
        "op": "remove"
    }
    servicebus_client = ServiceBusClient.from_connection_string(connection_string)
    with servicebus_client:
        receiver = servicebus_client.get_subscription_receiver(topic_name, subscription_name)
        with receiver:
            queue_len = 5
            while queue_len > 0:
                received_msgs = receiver.receive_messages(max_message_count=5, max_wait_time=5)
                queue_len = len(received_msgs)
                for msg in received_msgs:
                    try:
                        message = json.loads(str(msg))
                        if message:  # Check if a message is not empty
                            output_array = []
                            output_json = {}
                            try:
                                feed_service_api_url = str(url_first_part) + str(message["ProductId"]) + str(
                                    url_second_part)
                                print(str(message["ProductId"]))
                                if len(message["Description"]) > 150:
                                    if message["ProductType"] not in product_list:  # and all(attribute in message for attribute in noNER_attributes):
                                        product_description = message["Description"]
                                        output_json["value"] = summarization(product_description)
                                        output_json["path"] = "/AutoGeneratedDescription"
                                        output_json["op"] = "add"
                                        output_array.append(output_extra_json)
                                        output_array.append(output_json)
                                        response_data = json.dumps(output_array)
                                        requests.patch(feed_service_api_url, data=response_data)
                                        receiver.complete_message(msg)
                                        logger_tofile.info(
                                            "This product was successfully processed. It's not in 6 main attributes.")
                                    else:  # all(attribute in message for attribute in NER_attributes):
                                        product_description = message["Description"]
                                        product_name = message["ProductName"]
                                        product_type = message["ProductType"]
                                        attributes_list = message["Attributes"]
                                        target_name = create_target_name(attributes_list, product_type)
                                        if target_name:
                                            #print("I am a target name", target_name)
                                            summarized_description = summarization(product_description)
                                            #print(summarized_description)
                                            final_sentence = ner(summarized_description, product_name, target_name, product_type)
                                            #print(final_sentence)
                                            output_json["value"] = final_sentence
                                            output_json["path"] = "/AutoGeneratedDescription"
                                            output_json["op"] = "add"
                                            output_array.append(output_extra_json)
                                            output_array.append(output_json)
                                            response_data = json.dumps(output_array)
                                            #print(response_data)
                                            requests.patch(feed_service_api_url, data=response_data)
                                            receiver.complete_message(msg)
                                            logger_tofile.warning("This product was successfully processed. It's in 6 main attributes.")
                                        else:
                                            output_json["value"] = summarization(product_description)
                                            #print(output_json["value"])
                                            output_json["path"] = "/AutoGeneratedDescription"
                                            output_json["op"] = "add"
                                            output_array.append(output_extra_json)
                                            output_array.append(output_json)
                                            response_data = json.dumps(output_array)
                                            #print(response_data)
                                            requests.patch(feed_service_api_url, data=response_data)
                                            receiver.complete_message(msg)
                                            logger_tofile.info("This product was successfully processed but list of attributes was incomplete.")
                                else:
                                    output_array.append(output_extra_json)
                                    logger_tofile.info("Shorter than 150")
                                    response_data = json.dumps(output_array)
                                    requests.patch(feed_service_api_url, data=response_data)
                                    receiver.complete_message(msg)
                            except Exception as e:
                                print(e)
                                logger_tofile.error('List of attributes is corrupted', exc_info=sys.exc_info())
                                receiver.dead_letter_message(msg)
                        else:
                            logger_tofile.info("Something wrong with message")
                    except:
                        output_array.append(output_extra_json)
                        logger_tofile.error('Input JSON incomplete', exc_info=sys.exc_info())
                        receiver.dead_letter_message(msg)
        logs_to_blob()

if __name__ == '__main__':
    textprocessing()
