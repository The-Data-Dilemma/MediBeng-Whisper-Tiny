from fastapi import FastAPI, File, UploadFile, HTTPException
import tempfile
import os
from pydantic import BaseModel
from app.audio_processing import process_audio  # Correct import from app.audio_processing

app = FastAPI()

# Pydantic model for validation of the incoming file request
class AudioFile(BaseModel):
    file: UploadFile

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Check for file size or format restrictions
        if file.content_type not in ["audio/wav", "audio/mpeg", "audio/ogg"]:
            raise HTTPException(status_code=400, detail="Invalid file format. Only WAV, MP3, and OGG are supported.")
        
        # Save the uploaded audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            try:
                temp_audio.write(await file.read())
                temp_audio.close()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error saving the audio file: {str(e)}")

        # Check if the audio file is empty or corrupted
        if os.path.getsize(temp_audio.name) == 0:
            raise HTTPException(status_code=400, detail="The audio file is empty.")
        
        # Process the audio file and get the transcription
        transcription = process_audio(temp_audio.name)
        
        return {"transcription": transcription}

    except HTTPException as http_error:
        # If the error is a known HTTPException (e.g., file format or size error), return it
        raise http_error

    except Exception as e:
        # Return a general error message if any exception occurs
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
