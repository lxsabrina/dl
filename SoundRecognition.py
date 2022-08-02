# https://www.yiibai.com/ai_with_python/ai_with_python_speech_recognition.html
# https://www.bilibili.com/video/BV1yQ4y197hs/?vd_source=f31dcef75770e70f5bc0d4b7f3c83cee

import speech_recognition as sr
while True:
            r = sr.Recognizer()
            #启用麦克风
            mic = sr.Microphone()
            logging.info('录音中...')
            with mic as source:
                #降噪
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source)
            with open(f"00{wav_num}.wav", "wb") as f:
            #将麦克风录到的声音保存为wav文件
                f.write(audio.get_wav_data(convert_rate=16000))
            logging.info('录音结束，识别中...')
            target = audio_baidu(f"00{wav_num}.wav")
