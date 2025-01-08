import json,os
transcription = "datasets/librispeeh_asr/LibriSpeech/train-clean-100/19/198/19-198.trans.txt"
data = []
with open(transcription, 'r', encoding='utf-8') as file:
    for line in file:
        temp = {}
        lins = line.strip()
        temp['speech_path'] = os.path.join("/root/wangzhiyong/Freeze-Omni/datasets/librispeeh_asr/LibriSpeech/train-clean-100/19/198",lins[:11]+".flac")
        temp['transcription'] = lins[12:] 
        data.append(temp)

with open('utils/load_data/asrdata.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)