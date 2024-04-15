
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from pydub import AudioSegment
from fastapi import File, UploadFile
import os

from asr import AudioSpeechRecognition

app = FastAPI()
asr = None

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/asr")
async def audio_to_text(file: UploadFile = File(...)):
    file_name = file.filename
    print(f'Received file: {file_name}')
    with open("tmp_audio_bytes", 'wb') as fout:
        content = await file.read()
        fout.write(content)
    audio = AudioSegment.from_file("tmp_audio_bytes")
    audio = audio.set_frame_rate(16000)
    # bytes to wav
    file_name = file_name +'.wav'
    audio.export(f"{file_name}", format="wav")
    try:
        asr_result = asr.audio2text(file_name)
    except Exception as e:
        print(e)
        asr_result = e
    finally:
        os.remove(file_name)
        os.remove("tmp_audio_bytes")
    return {"asr_result": asr_result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--model_name_or_path", type=str, default="openai/whisper-tiny")
    parser.add_argument("--bf16", default=False, action="store_true")
    parser.add_argument("--language", type=str, default="auto")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    asr = AudioSpeechRecognition(
        model_name_or_path=args.model_name_or_path,
        bf16=args.bf16,
        language=args.language,
        device=args.device
    )

    uvicorn.run(app, host=args.host, port=args.port)