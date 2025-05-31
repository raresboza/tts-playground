import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

def main():
    model = ChatterboxTTS.from_pretrained(device="cuda")

    text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
    wav = model.generate(text)
    ta.save("output/test-1.wav", wav, model.sr)

if __name__ == "__main__":
    main()